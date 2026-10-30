"""Frozen real-log normalization audit. Never runs games or timed searches."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import copy
from dataclasses import asdict
from dataclasses import replace
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import re
from statistics import median
import sys
from time import perf_counter
from types import SimpleNamespace, ModuleType
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_X = '64dd15b53bc755af443cee49948c368ce41bac3097a3eab32c5c95a6875e68e3'
REFERENCE_SEARCH = '09142bcc462f3ae8954e6527c21cb4f80784e1d0c4db2b10372db438a44d3dc4'
FROZEN_SOURCES = ROOT / 'tests/fixtures/deep_v7_sources.zip'
OBSERVED_FIELDS = ('game_id', 'board', 'side', 'remaining_moves', 'rows', 'cols',
                   'board_size', 'score_1', 'score_2', 'multiplier_1', 'multiplier_2')
CONTEXT_FIELDS = ('side', 'head', 'enemy_head', 'scores', 'multipliers', 'food',
                  'numbered_food', 'next_food_digit', 'pickups', 'walls',
                  'crash_history', 'tail_food_a', 'tail_food_b', 'remaining_moves',
                  'current_target', 'hunger', 'repeat_count', 'plan', 'direction',
                  'reason', 'legal', 'active_games', 'search_time_budget_ms', 'commit_to_plan')


def digest(data):
    return hashlib.sha256(data).hexdigest()


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True))


def manifest_names(report):
    section = Path(report).read_text(encoding='utf-8').split('## Manifiesto de la tanda fija', 1)[1]
    names = sorted(set(re.findall(r'game_[a-f0-9-]+\.log', section)))
    if len(names) != 86:
        raise ValueError('Expected frozen 86-file report manifest')
    return names


def load_cohort(report, frozen_manifest, games):
    frozen = json.loads(Path(frozen_manifest).read_text())
    names = manifest_names(report)
    if names != sorted(frozen['filenames']) or digest(Path(report).read_bytes()) != frozen['report_sha256']:
        raise ValueError('Frozen report/manifest mismatch')
    cases = []
    for name in names:
        path = Path(games) / name
        if digest(path.read_bytes()) != frozen['log_hashes'][name]:
            raise ValueError('Frozen log hash mismatch: ' + name)
        cases.extend(read_log(path))
    return cases


def read_log(path):
    path = Path(path)
    raw = path.read_bytes()
    log_hash = digest(raw)
    observed, observed_line, version = None, None, None
    history, cases = [], []
    for number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
        if not line.startswith(('= ', '< ', '? ')):
            continue
        event = json.loads(line[2:])
        if event.get('event') == 'bot_version':
            version = event.get('version')
        if event.get('event') == 'your_turn':
            observed = {k: event['data'][k] for k in OBSERVED_FIELDS if k in event['data']}
            observed_line = number
        if event.get('event') != 'decision_debug':
            continue
        # verification.after is the CURRENT turn, not the sent-action prediction.
        snapshot = event.get('verification', {}).get('after') or {}
        context = {k: event[k] for k in CONTEXT_FIELDS if k in event}
        context['tracked_body'] = snapshot.get('tracked_body', {})
        diagnostic = dict(context)
        for field in ('deep_moves', 'food_options', 'candidate_moves', 'commit_to_plan'):
            if field in event:
                diagnostic[field] = event[field]
        case = {'provenance': {'filename': path.name, 'game_id': event['game_id'],
                              'debug_line': number, 'observed_line': observed_line,
                              'log_sha256': log_hash, 'debug_sha256': digest(line.encode()),
                              'declared_version': version, 'source': 'real_server_log'},
                'observed': observed, 'diagnostic': diagnostic,
                'history': list(history[-80:])}
        cases.append(case)
        history.append({'observed': observed, 'observed_line': observed_line,
                        'debug_line': number, 'context': context})
    return cases


def terminal_class(values, remaining, depth):
    if remaining is not None and 0 <= remaining <= depth:
        return 'terminal-horizon'
    if any(abs(v) >= 300_000_000 for v in values):
        return 'terminal-compatible/extreme'
    return 'normal'


def legacy_adjustments(rows, depth, percent=35, cap=25000, minimum_depth=4):
    result = {r['direction']: 0 for r in rows}
    usable = [r for r in rows if r.get('search_score') is not None and not r.get('forced_loss')]
    if depth < minimum_depth or not usable:
        return result
    clipped = {r['direction']: max(-200000, min(200000, r['search_score'])) for r in usable}
    center = sum(clipped.values()) // len(clipped)
    result.update({key: max(-cap, min(cap, (value - center) * percent // 100))
                   for key, value in clipped.items()})
    return result


def hypothetical_adjustments(rows, depth, remaining, root_terminal=False):
    """Approved formula for offline measurement only; never installed in a brain."""
    usable = [r for r in rows if r.get('search_score') is not None and not r.get('forced_loss')]
    guarded = root_terminal or (remaining is not None and remaining <= depth)
    guarded |= any(abs(r['search_score']) >= 300000000 for r in usable)
    if guarded or depth < 4 or not usable:
        return legacy_adjustments(rows, depth)
    center = sum(r['search_score'] for r in usable) // len(usable)
    result = {r['direction']: 0 for r in rows}
    result.update({r['direction']: max(-25000, min(25000, (r['search_score'] - center) * 35 // 100))
                   for r in usable})
    return result


def load_baseline(path):
    return load_source(path, REFERENCE_SEARCH)


def frozen_source(name, expected_hash=None):
    """Read exact historical bytes, independently of mutable production code."""
    with zipfile.ZipFile(FROZEN_SOURCES) as archive:
        manifest = json.loads(archive.read('provenance.json'))
        raw = archive.read(name)
    if digest(raw) != (expected_hash or manifest['sources'][name]):
        raise ValueError('Frozen historical source hash mismatch: ' + name)
    return raw


def load_frozen_module(name, expected_hash):
    raw = frozen_source(name, expected_hash)
    module_name = '_historical_' + expected_hash[:12]
    module = ModuleType(module_name)
    module.__file__ = str(FROZEN_SOURCES) + '/' + name
    sys.modules[module_name] = module
    exec(compile(raw, module.__file__, 'exec'), module.__dict__)
    return module


def load_reference_planner(weights):
    planner = load_frozen_module('snake_bot/food_planner.py', REFERENCE_X)
    planner.get_weights = lambda: weights
    return planner


def load_source(path, expected_hash):
    path = Path(path)
    raw = path.read_bytes()
    if digest(raw) != expected_hash:
        raise ValueError('Frozen move_search baseline hash mismatch')
    name = '_frozen_move_search_' + expected_hash[:12]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(raw, str(path), 'exec'), module.__dict__)
    return module


def rerank_recorded(case, source, candidate=False, override=None):
    d = case['diagnostic']
    original = d.get('deep_moves', [])
    order = {direction: i for i, direction in enumerate(d.get('legal', []))}
    rows = sorted(copy.deepcopy(original), key=lambda r: order.get(r['direction'], 99))
    for r in rows:
        r['score'] -= r.get('search_adjustment', 0)
    rows.sort(key=source.deep_sort_key, reverse=True)
    depth = max((r.get('search_depth', 0) for r in rows), default=0)
    root_terminal = d.get('remaining_moves', 1) <= 0 or any(v < -2500 for v in d['scores'].values())
    adjustments = (hypothetical_adjustments(rows, depth, d.get('remaining_moves'), root_terminal)
                   if candidate else legacy_adjustments(rows, depth))
    if override is not None:
        adjustments = override
    for r in rows:
        r['search_adjustment'] = adjustments[r['direction']]
        r['score'] += adjustments[r['direction']]
    rows.sort(key=source.deep_sort_key, reverse=True)
    source.prefer_safer_close_move(rows)
    return rows


def measure_hypothesis(cases, source):
    counts, games, gaps, examples = Counter(), defaultdict(set), defaultdict(list), []
    for case in cases:
        a = analyze_case(case)
        d = case['diagnostic']
        rows = d.get('deep_moves', [])
        old_rank = rerank_recorded(case, source)
        counts['legacy_order_control_mismatch'] += old_rank != rows
        if not a['eligible_depth_and_two']:
            continue
        counts['eligible_depth_and_two'] += 1
        usable = sorted((r for r in rows if r.get('search_score') is not None and not r.get('forced_loss')),
                        key=lambda r: r['search_score'], reverse=True)
        root_terminal = d.get('remaining_moves', 1) <= 0 or any(v < -2500 for v in d['scores'].values())
        guarded = root_terminal or d['remaining_moves'] <= a['depth'] or any(abs(r['search_score']) >= 300000000 for r in usable)
        old = legacy_adjustments(rows, a['depth'])
        capped = hypothetical_adjustments(rows, a['depth'], d['remaining_moves'], root_terminal)
        if guarded:
            counts['guarded_legacy_whole_group'] += 1
            counts['guarded_mapping_changed'] += capped != old
            continue
        counts['normal_eligible'] += 1
        counts['normal_differing_raw'] += a['eligible_differing_raw']
        center = sum(r['search_score'] for r in usable) // len(usable)
        uncapped = {r['direction']: (r['search_score'] - center) * 35 // 100 for r in usable}
        uncapped.update({r['direction']: 0 for r in rows if r['direction'] not in uncapped})
        new_rank = rerank_recorded(case, source, candidate=True)
        uncap_rank = rerank_recorded(case, source, override=uncapped)
        old_order = [r['direction'] for r in old_rank]
        new_order = [r['direction'] for r in new_rank]
        uncap_order = [r['direction'] for r in uncap_rank]
        first, second = usable[:2]
        u1, u2 = uncapped[first['direction']], uncapped[second['direction']]
        c1, c2 = capped[first['direction']], capped[second['direction']]
        flags = {
            'adjustment_changed': capped != old,
            'order_changed': new_order != old_order,
            'top_ranked_move_changed': new_order[0] != old_order[0],
            'uncapped_order_changed': uncap_order != old_order,
            'uncapped_mapping_changed': uncapped != old,
            'normal_old_flat_loss': a['all_clip_identical'],
            'normal_old_flat_recovered_adjustment': a['all_clip_identical'] and capped != old,
            'normal_old_flat_recovered_order': a['all_clip_identical'] and new_order != old_order,
            'new_final_cap_active': any(abs(v) > 25000 for v in uncapped.values()),
            'cap_a_top1_top2_gap_lost': u1 != u2 and c1 == c2,
            'cap_b_uncapped_order_changes_but_capped_stays_old': uncap_order != old_order and new_order == old_order,
            'cap_c_uncapped_mapping_changes_but_capped_stays_old': uncapped != old and capped == old,
        }
        counts['raw_top1_top2_positive_gap'] += a['raw_gap'] > 0
        counts['uncapped_top1_top2_positive_gap'] += u1 != u2
        for key, value in flags.items():
            counts[key] += value
            if value:
                games[key].add(case['provenance']['game_id'])
        gaps['normal_raw_top1_top2'].append(a['raw_gap'])
        gaps['normal_uncapped_top1_top2'].append(u1-u2)
        gaps['normal_capped_top1_top2'].append(c1-c2)
        if a['eligible_differing_raw']:
            gaps['normal_differing_raw_top1_top2'].append(a['raw_gap'])
        if flags['order_changed'] or any(flags[key] for key in flags if key.startswith('cap_')):
            examples.append({'provenance': case['provenance'], 'raw_gap': a['raw_gap'], 'raw_top1': a['raw_top1'],
                             'raw_top2': a['raw_top2'], 'old_adjustments': old, 'uncapped': uncapped, 'capped': capped,
                             'old_order': old_order, 'uncapped_order': uncap_order, 'capped_order': new_order,
                             'flags': flags})
    denominator = counts['normal_eligible']
    return {'formula': 'applicable non-forced raw integer mean -> 35% floor -> +/-25000 cap; whole old group if applicable extreme >=300M, R<=depth or root terminal',
            'scope': 'hypothetical frozen historical shallow bases; no candidate brain or competitive benefit claim',
            'counts': dict(counts), 'games_with_at_least_one': {k: len(v) for k, v in games.items()},
            'cap_definitions_and_denominators': {
                'a': {'definition': 'uncapped top1/top2 adjustments differ but capped adjustments equal',
                      'normal_pool': denominator, 'uncapped_positive_top1_top2_gap': counts['uncapped_top1_top2_positive_gap']},
                'b': {'definition': 'uncapped full order differs from old; capped full order equals old, after unchanged sort/prefer',
                      'normal_pool': denominator, 'uncapped_order_changed': counts['uncapped_order_changed']},
                'c': {'definition': 'uncapped adjustment map differs from old; capped map equals old',
                      'normal_pool': denominator, 'uncapped_mapping_changed': counts['uncapped_mapping_changed']}},
            'gap_distributions': {k: distribution(v) for k, v in gaps.items()},
            'top_examples': sorted(examples, key=lambda r: r['raw_gap'], reverse=True)[:12],
            'cap_examples': [r for r in examples if any(r['flags'][key] for key in r['flags'] if key.startswith('cap_'))][:8]}


def state_for(case):
    from snake_bot.snake_state import parse_state, legal_moves, step
    d = case['diagnostic']
    state = parse_state(case['observed'], crash_history=d.get('crash_history'))
    tracked = d['tracked_body']
    if set(tracked) != {'A', 'B'}:
        raise ValueError('Missing real tracked bodies')
    bodies = {side: tuple(map(tuple, tracked[side]['body'])) for side in ('A', 'B')}
    if any(set(bodies[side]) != set(state.body(side)) for side in ('A', 'B')):
        raise ValueError('Tracked body cells disagree with board')
    state = replace(state, snakes=bodies,
                    reliable_tails=frozenset(side for side in ('A', 'B') if tracked[side]['order_reliable']))
    fields = {'side': state.side, 'head': state.head(state.side), 'enemy_head': state.head(state.enemy),
              'remaining_moves': state.remaining_moves, 'scores': state.scores,
              'multipliers': state.multipliers, 'food': sorted(state.food),
              'numbered_food': sorted(state.food_values.items()), 'next_food_digit': state.next_food_digit,
              'pickups': sorted(state.pickups), 'walls': sorted(state.walls),
              'tail_food_a': sorted(state.tail_food_a), 'tail_food_b': sorted(state.tail_food_b)}
    for name, value in fields.items():
        if normalized(value) != normalized(d[name]):
            raise ValueError('Recorded state mismatch: ' + name)
    legal = legal_moves(state, state.side)
    clean = [m for m in legal if step(state.head(state.side), m) not in state.wrong_food()
             and step(state.head(state.side), m) not in state.walls]
    if set(clean or legal) != set(d['legal']):
        raise ValueError('Recorded legal moves mismatch')
    return state, {'observable_state_exact': True, 'tracked_body_order_from_log': True,
                   'history_counters_from_log': True,
                   'unlogged_internal_cache_history_certified': False}


def compact_fixture_history(case, signature_cache=None):
    from snake_bot.snake_brain import SnakeBrain
    from snake_bot.food_planner import should_use_plan_for_positioning
    if isinstance(case['history'], dict):
        compacted = copy.deepcopy(case)
        if compacted['history'].get('signature_format') != 'sha256-canonical-json':
            compacted['history']['signatures'] = [digest(json.dumps(s, sort_keys=True, separators=(',', ':')).encode()) for s in compacted['history']['signatures']]
            compacted['history']['signature_format'] = 'sha256-canonical-json'
        return compacted
    brain = SnakeBrain()
    game_id = case['provenance']['game_id']
    lines = None
    provenance = []
    signatures = []
    for entry in case['history']:
        context = entry['context']
        identity = (game_id, entry['debug_line'])
        if signature_cache is not None and identity in signature_cache:
            signature, trace = signature_cache[identity]
            signatures.append(signature)
            provenance.append(trace)
            continue
        if not context.get('tracked_body') or not context.get('current_target') and 'hunger' not in context:
            continue
        state, _ = state_for({'observed': entry['observed'], 'diagnostic': context})
        if 'commit_to_plan' in context:
            recorded = context
        else:
            if lines is None:
                lines = (ROOT / 'games' / case['provenance']['filename']).read_text(encoding='utf-8-sig').splitlines()
            recorded = json.loads(lines[entry['debug_line'] - 1][2:])
        plan = context.get('plan')
        if context['reason'] == 'advantage_control':
            target = None
        elif plan and recorded.get('commit_to_plan') and plan['first_move'] in context['legal']:
            target = tuple(plan['food'])
        elif plan and should_use_plan_for_positioning(SimpleNamespace(**plan)):
            target = tuple(plan['food'])
        else:
            target = None
        brain.targets[(game_id, state.side)] = target
        brain._remember(game_id, state)
        signature = digest(json.dumps(normalized(brain.recent_states[game_id][-1]), sort_keys=True, separators=(',', ':')).encode())
        trace = {'debug_line': entry['debug_line'], 'observed_line': entry['observed_line'],
                 'post_target': target, 'derivation': 'recorded branch plus unchanged target assignment'}
        signatures.append(signature)
        provenance.append(trace)
        if signature_cache is not None:
            signature_cache[identity] = (signature, trace)
    compacted = dict(case)
    compacted['history'] = {'signatures': signatures, 'signature_format': 'sha256-canonical-json',
                            'signature_provenance': provenance,
                            'full_history_sha256': digest(json.dumps(case['history'], sort_keys=True).encode()),
                            'external_history_key': game_id + ':' + str(case['provenance']['debug_line'])}
    return compact_fixture_history(compacted)


def hydrate_brain(case, brain):
    from collections import deque
    state, verification = state_for(case)
    d = case['diagnostic']
    game_id, side = case['provenance']['game_id'], state.side
    key = (game_id, side)
    brain.targets[key] = tuple(d['current_target']) if d.get('current_target') is not None else None
    brain.turns_since_food[key] = d.get('hunger', 0) - 1
    brain.last_scores[key] = state.scores[side]
    brain.current_states[game_id] = state
    brain.observed_states[game_id] = state
    brain.observation_tokens[game_id] = (None, state.remaining_moves)
    brain.previous_snakes[game_id] = state.snakes
    brain.crash_history[game_id] = dict(d['crash_history'])
    # These keys encode only occupancy count; no fabricated boards are used.
    for index in range(max(0, d.get('active_games', 1) - 1)):
        brain.previous_snakes['_replay_occupancy_' + str(index)] = None
    brain.recent_states[game_id] = deque(case['history']['signatures'], maxlen=80)
    repeat = brain._repeat_count(game_id, state)
    if repeat != d.get('repeat_count', 0):
        raise ValueError(f'Recorded repeat history mismatch: {repeat} != {d.get("repeat_count")}')
    return state, verification


def hashed_repeat_count(self, game_id, state):
    from snake_bot.snake_brain import SnakeBrain
    probe = SnakeBrain()
    probe.targets[(game_id, state.side)] = self.targets.get((game_id, state.side))
    probe._remember(game_id, state)
    signature = digest(json.dumps(normalized(probe.recent_states[game_id][0]), sort_keys=True, separators=(',', ':')).encode())
    return sum(value == signature for value in self.recent_states.get(game_id, ()))


def replay_current_reference(case, source, candidate=None):
    from snake_bot import snake_brain, bot_weights
    from tools.compare_food_planners import private_namespace, private_class
    weights = dict(bot_weights.get_weights())
    # This is a historical replay: v8/v9 source changes must not silently
    # replace the exact accepted X planner or force its hash to be repinned.
    food_planner = load_reference_planner(weights)
    tape = FrozenEngine(case)
    def brain_factory(ranking_source):
        ranking = private_namespace(ranking_source, food_planner, weights)
        cache_class = private_class(ranking_source.TurnSearchCache, ranking)
        ranking['TurnSearchCache'] = cache_class
        ranking['IterativeSearchEngine'] = tape
        namespace = private_namespace(snake_brain, food_planner, weights)
        namespace['TurnSearchCache'] = cache_class
        namespace['rank_deep_moves'] = ranking['rank_deep_moves']
        brain_class = private_class(snake_brain.SnakeBrain, namespace)
        brain_class._repeat_count = hashed_repeat_count
        return brain_class, namespace
    d = case['diagnostic']
    record = {'provenance': case['provenance'], 'historic_direction': d['direction'],
              'historic_plan': d.get('plan'), 'comparison': 'current X64 legacy baseline versus historic 2.16 pre-X log'}
    brains, results, frozen_plans = [], [], None
    for ranking_source in (source, source if candidate is None else candidate):
        brain_class, namespace = brain_factory(ranking_source)
        brain = brain_class(enable_debug=True)
        try:
            state, verification = hydrate_brain(case, brain)
        except ValueError as error:
            return dict(record, status='unsupported_context', exclusion=str(error))
        hunger = d.get('hunger', 0)
        target = brain.targets[(case['provenance']['game_id'], state.side)]
        if frozen_plans is None:
            frozen_plans = food_planner.build_food_plans(state, state.side, target, hunger)
        plans = frozen_plans
        def cached_plans(actual_state, actual_side, actual_target, actual_hunger):
            if actual_state != state or actual_side != state.side or actual_target != target or actual_hunger != hunger:
                raise ValueError('Planner inputs differ between replay versions')
            return list(frozen_plans)
        namespace['build_food_plans'] = cached_plans
        plan = food_planner.choose_food_plan(state, state.side, brain.targets[(case['provenance']['game_id'], state.side)], hunger, plans=plans)
        plan_debug = brain._plan_debug(plan)
        record['current_x_plan'] = normalized(plan_debug)
        if normalized(None if plan is None else plan.food) != (d.get('plan') or {}).get('food'):
            return dict(record, status='unsupported_changed_target', exclusion='old root score target differs from current X64 target; no retimed search authorized')
        if not d.get('deep_moves'):
            return dict(record, status='unsupported_no_search_result', exclusion='no historical SearchResult in decision')
        try:
            move = brain.choose_move(copy.deepcopy(case['observed']))
            final = brain.safe_direction(case['observed'], move)
        except ValueError as error:
            return dict(record, status='unsupported_search_request', exclusion=str(error))
        debug = brain.decision_debug(case['provenance']['game_id'])
        brains.append(brain)
        results.append({'choose_direction': move, 'final_direction': final, 'debug': copy.deepcopy(debug)})
    first, control = results
    raw = {r['direction']: r.get('search_score') for r in first['debug'].get('deep_moves', []) if r.get('search_score') is not None}
    historic_rows = d.get('deep_moves', [])
    current_rows = first['debug'].get('deep_moves', [])
    historic_map = {r['direction']: r for r in historic_rows}
    def base_map(debug):
        result = {}
        for row in debug.get('deep_moves', []):
            row = dict(row)
            row['score'] -= row.pop('search_adjustment', 0)
            result[row['direction']] = row
        return result
    shallow_exact = base_map(first['debug']) == base_map(control['debug'])
    result = dict(record, status='controlled_historical_raw_replay',
                verification=verification, request=tape.request, search_calls=tape.calls,
                raw_result_preserved=raw == tape.result.scores,
                shared_request_and_result_exact=shallow_exact and tape.calls == 2,
                shallow_bases_and_info_exact=shallow_exact,
                historical_request_certified=False,
                historical_request_unknowns=['unlogged max_depth', 'unlogged full compact state/target request', 'unlogged process weights/source hashes'],
                repeat_count_exact=first['debug'].get('repeat_count') == d.get('repeat_count'),
                hunger_exact=first['debug'].get('hunger') == d.get('hunger'),
                safe_direction_applied=True, safe_repair=first['choose_direction'] != first['final_direction'],
                current_x_final_direction=first['final_direction'], control_final_direction=control['final_direction'],
                historic_vs_current_x_final_changed=d['direction'] != first['final_direction'],
                current_x_plan_changed=normalized(d.get('plan')) != normalized(first['debug'].get('plan')),
                historic_vs_current_x_adjustment_changed=any(r.get('search_adjustment') != historic_map.get(r['direction'], {}).get('search_adjustment') for r in current_rows),
                historic_vs_current_x_order_changed=[r['direction'] for r in historic_rows] != [r['direction'] for r in current_rows],
                current_x_debug=first['debug'])
    if candidate is not None:
        candidate_rows = control['debug'].get('deep_moves', [])
        current_map = {r['direction']: r for r in current_rows}
        candidate_raw = {r['direction']: r['search_score'] for r in candidate_rows if r.get('search_score') is not None}
        result.update(candidate_final_direction=control['final_direction'],
                      candidate_final_changed=control['final_direction'] != first['final_direction'],
                      candidate_choose_changed=control['choose_direction'] != first['choose_direction'],
                      candidate_safe_repair=control['choose_direction'] != control['final_direction'],
                      candidate_adjustment_changed=any(r['search_adjustment'] != current_map[r['direction']]['search_adjustment'] for r in candidate_rows),
                      candidate_order_changed=[r['direction'] for r in current_rows] != [r['direction'] for r in candidate_rows],
                      candidate_top_ranked_changed=current_rows[0]['direction'] != candidate_rows[0]['direction'],
                      raw_result_preserved=result['raw_result_preserved'] and candidate_raw == tape.result.scores,
                      candidate_debug=control['debug'])
    return result


def replay_comparison(case, baseline, candidate):
    return replay_current_reference(case, baseline, candidate)


def comparison_summary(records, eligible_total, manifest, guard_controls):
    counts = Counter(eligible_historical_pool=eligible_total, processed=len(records),
                     supported_common_input=0, historical_request_certified=0)
    games = defaultdict(set)
    flags = ('candidate_adjustment_changed', 'candidate_order_changed', 'candidate_top_ranked_changed',
             'candidate_final_changed', 'candidate_choose_changed', 'safe_repair', 'candidate_safe_repair',
             'historic_vs_current_x_final_changed', 'historic_vs_current_x_adjustment_changed',
             'historic_vs_current_x_order_changed', 'raw_result_preserved', 'shared_request_and_result_exact',
             'shallow_bases_and_info_exact', 'repeat_count_exact', 'hunger_exact')
    for record in records:
        status = record['status']
        if status == 'controlled_historical_raw_replay':
            counts['supported_common_input'] += 1
        else:
            counts[status] += 1
        for flag in flags:
            counts[flag] += bool(record.get(flag))
            if record.get(flag):
                games[flag].add(record['provenance']['game_id'])
    examples = [r for r in records if r.get('candidate_final_changed')][:15]
    exclusions = [r for r in records if r['status'] != 'controlled_historical_raw_replay']
    return {'counts': dict(counts), 'games_with_at_least_one': {k: len(v) for k, v in games.items()},
            'guard_controls': guard_controls, 'timed_searches': 0,
            'manifest': manifest, 'final_move_examples': examples, 'exclusions': exclusions,
            'interpretation': 'Actual current-X64 brain before/after followed by safe_direction; historical raw target matches required; unlogged historical request/source not certified; one-step control, no game counterfactual or competitive benefit claim'}


def run_comparison(cases, baseline, candidate, output, limit=None):
    from snake_bot.bot_weights import get_weights
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    eligible = [c for c in cases if analyze_case(c)['eligible_depth_and_two']]
    if limit is not None:
        eligible = eligible[:limit]
    records, signature_cache, guarded, guard_mismatches = [], {}, 0, 0
    started = perf_counter()
    with gzip.open(output / 'comparison_rows.jsonl.gz', 'xt', encoding='utf-8') as stream:
        for index, case in enumerate(eligible, 1):
            a = analyze_case(case)
            if a['terminal_class'] != 'normal':
                state, _ = state_for(case)
                guarded += 1
                actual = candidate._search_adjustments(state, case['diagnostic']['deep_moves'], a['depth'], get_weights())
                old = legacy_adjustments(case['diagnostic']['deep_moves'], a['depth'])
                guard_mismatches += any(actual.get(k, 0) != v for k, v in old.items())
            try:
                compacted = compact_fixture_history(case, signature_cache)
                record = replay_comparison(compacted, baseline, candidate)
            except ValueError as error:
                record = {'provenance': case['provenance'], 'status': 'unsupported_context', 'exclusion': str(error)}
            stream.write(json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n')
            if record['status'] == 'controlled_historical_raw_replay' and not all(record[k] for k in ('raw_result_preserved', 'shared_request_and_result_exact', 'shallow_bases_and_info_exact', 'hunger_exact', 'repeat_count_exact')):
                raise ValueError('Replay invariant failed: ' + str(record['provenance']))
            small = {k: v for k, v in record.items() if k not in ('current_x_debug', 'candidate_debug', 'request')}
            if record.get('candidate_final_changed'):
                small['before_reason'] = record['current_x_debug']['reason']
                small['after_reason'] = record['candidate_debug']['reason']
            records.append(small)
            if index % 100 == 0 or index == len(eligible):
                stream.flush()
                supported = sum(r['status'] == 'controlled_historical_raw_replay' for r in records)
                changed = sum(bool(r.get('candidate_final_changed')) for r in records)
                elapsed = perf_counter() - started
                eta = elapsed / index * (len(eligible) - index)
                print(json.dumps({'processed': index, 'total': len(eligible), 'supported': supported,
                                  'final_changed': changed, 'elapsed_seconds': round(elapsed, 1),
                                  'eta_seconds': round(eta, 1)}), flush=True)
    source_manifest = {'baseline_move_search_sha256': digest(Path(baseline.__file__).read_bytes()),
                       'candidate_move_search_sha256': digest(Path(candidate.__file__).read_bytes()),
                       'planner_x_sha256': REFERENCE_X, 'engine_sha256': digest((ROOT / 'snake_bot/search_engine.py').read_bytes()),
                       'weights_sha256': digest((ROOT / 'weights/active_weights.json').read_bytes()),
                       'normalization_percent': get_weights()['iterative_search_weight_percent'],
                       'normalization_cap': get_weights()['iterative_search_adjustment_cap'],
                       'cohort_log_hashes': {c['provenance']['filename']: c['provenance']['log_sha256'] for c in cases}}
    result = comparison_summary(records, len(eligible), source_manifest,
                                {'historical_guarded_cases': guarded, 'actual_helper_legacy_mapping_mismatches': guard_mismatches})
    write_new(output / 'comparison.json', result)
    text = '# Current X64 exact common-input one-step replay\n\n' + result['interpretation'] + '\n\n```json\n' + json.dumps({k: v for k, v in result.items() if k not in ('exclusions', 'final_move_examples')}, indent=2) + '\n```\n\n## Final Move Examples\n\n'
    for row in result['final_move_examples']:
        p = row['provenance']
        text += f"- {p['filename']}:{p['debug_line']}: {row['current_x_final_direction']} -> {row['candidate_final_direction']} ({row['before_reason']} -> {row['after_reason']})\n"
    text += '\nComplete rows and shared requests: comparison_rows.jsonl.gz. Exclusions: comparison.json.\n'
    write_new(output / 'comparison.md', text)
    return result


class FrozenEngine:
    """Transport full historical results, audit requests, never call search()."""
    def __init__(self, case):
        from snake_bot.search_engine import SearchResult
        self.case = case
        rows = case['diagnostic'].get('deep_moves', [])
        first = rows[0] if rows else {}
        self.result = SearchResult(
            scores={r['direction']: r['search_score'] for r in rows if r.get('search_score') is not None},
            completed_depth=first.get('search_depth', 0), nodes=first.get('search_nodes', 0),
            transposition_hits=first.get('search_transposition_hits', 0), elapsed_ms=first.get('search_elapsed_ms', 0))
        self.request = None
        self.calls = 0

    def from_game_state(self, state, time_budget_ms=90, max_depth=6):
        from snake_bot.search_engine import IterativeSearchEngine
        expected, _ = state_for(self.case)
        _, compact = IterativeSearchEngine.from_game_state(state, time_budget_ms, max_depth)
        _, expected_compact = IterativeSearchEngine.from_game_state(expected, time_budget_ms, max_depth)
        if compact != expected_compact:
            raise ValueError('Recorded state mismatch for frozen search')
        def search(actual_compact, side, legal, target=None):
            request = normalized({'state': asdict(actual_compact), 'side': side,
                                  'legal': list(legal), 'target': target,
                                  'budget': time_budget_ms, 'max_depth': max_depth})
            if self.request is not None and request != self.request:
                raise ValueError('Frozen request mismatch between replay versions')
            historical_target = (self.case['diagnostic'].get('plan') or {}).get('food')
            if (request['state'] != normalized(asdict(expected_compact))
                    or side != expected.side
                    or set(legal) != set(self.case['diagnostic']['legal'])
                    or normalized(target) != historical_target
                    or time_budget_ms != self.case['diagnostic'].get('search_time_budget_ms')):
                raise ValueError('Historical request mismatch: target/legal/budget/state')
            self.request = request
            self.calls += 1
            return copy.deepcopy(self.result)
        return SimpleNamespace(search=search), compact


def analyze_case(case):
    d = case['diagnostic']
    rows = d.get('deep_moves', [])
    depth = max((r.get('search_depth', 0) for r in rows), default=0)
    usable = [r for r in rows if r.get('search_score') is not None and not r.get('forced_loss')]
    values = sorted((r['search_score'] for r in usable), reverse=True)
    all_values = [r['search_score'] for r in rows if r.get('search_score') is not None]
    clipped = [max(-200000, min(200000, v)) for v in values]
    eligible = depth >= 4 and len(values) >= 2
    differing = eligible and len(set(values)) >= 2
    adjustments = legacy_adjustments(rows, depth)
    center = sum(clipped) // len(clipped) if clipped else 0
    uncapped = [(v - center) * 35 // 100 for v in clipped]
    capped = [max(-25000, min(25000, v)) for v in uncapped]
    pairs = [(i, j) for i in range(len(values)) for j in range(i + 1, len(values))]
    geometry = 'no_raw'
    if values:
        if all(v > 200000 for v in values):
            geometry = 'positive_above_200k'
        elif all(v < -200000 for v in values):
            geometry = 'negative_below_minus200k'
        elif all(abs(v) <= 200000 for v in values):
            geometry = 'inside_200k'
        else:
            geometry = 'inside_outside_or_opposite'
    return {'eligible_depth_and_two': eligible, 'eligible_differing_raw': differing,
            'depth': depth, 'raw_top1': values[0] if values else None,
            'raw_top2': values[1] if len(values) >= 2 else None,
            'raw_gap': values[0] - values[1] if len(values) >= 2 else None,
            'raw_spread': max(values) - min(values) if values else None,
            'terminal_class': terminal_class(all_values, d.get('remaining_moves'), depth),
            'extreme_sources_compatible': sorted(set(
                ('elimination_500m' if abs(v) >= 400000000 else 'final_300m') +
                ('_positive' if v > 0 else '_negative')
                for v in all_values if abs(v) >= 300000000)),
            'geometry': geometry, 'opposite_signs': bool(values and min(values) < 0 < max(values)),
            'inside_and_outside': any(abs(v) <= 200000 for v in values) and any(abs(v) > 200000 for v in values),
            'all_clip_identical': differing and len(set(clipped)) == 1,
            'forced_loss_present': any(r.get('forced_loss') for r in rows),
            'SAFE2': sum(r.get('safety') == 'SAFE' and not r.get('forced_loss') for r in usable) >= 2,
            'old_adjustments': adjustments,
            'actual_adjustments_match': all(r.get('search_adjustment', 0) == adjustments[r['direction']] for r in rows),
            'clip_masked_pairs': sum(values[i] != values[j] and clipped[i] == clipped[j] for i, j in pairs) if eligible else 0,
            'final_cap_masked_pairs': sum(uncapped[i] != uncapped[j] and capped[i] == capped[j] for i, j in pairs) if eligible else 0,
            'old_cap_active': eligible and any(abs(v) > 25000 for v in uncapped),
            'old_floor_masked_pairs': sum(clipped[i] != clipped[j] and uncapped[i] == uncapped[j] for i, j in pairs) if eligible else 0}


def distribution(values):
    values = sorted(values)
    if not values:
        return {'n': 0, 'min': None, 'median': None, 'p90': None, 'p95': None, 'max': None, 'histogram': {}}
    bins = Counter('0' if v == 0 else '1..100' if v <= 100 else '101..1000' if v <= 1000
                   else '1001..10000' if v <= 10000 else '10001..100000' if v <= 100000
                   else '100001..1000000' if v <= 1000000 else '>1000000' for v in values)
    return {'n': len(values), 'min': values[0], 'median': median(values),
            'p90': values[(len(values)-1)*90//100], 'p95': values[(len(values)-1)*95//100],
            'max': values[-1], 'histogram': dict(sorted(bins.items()))}


def summarize(cases):
    counts, by_class, games = Counter(), defaultdict(Counter), defaultdict(set)
    gaps, differing_gaps, spreads, examples = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
    observed_terminal_categories = Counter()
    analyzed = []
    for case in cases:
        a = analyze_case(case)
        analyzed.append((case, a))
        counts['decisions'] += 1
        counts['actual_adjustment_mismatch'] += not a['actual_adjustments_match']
        counts['forced_loss_all_decisions'] += a['forced_loss_present']
        observed_terminal_categories[a['terminal_class']] += 1
        for marker in a['extreme_sources_compatible']:
            observed_terminal_categories[marker] += 1
        if not a['eligible_depth_and_two']:
            continue
        cls = a['terminal_class']
        keys = ['eligible_depth_and_two', a['geometry']]
        for flag in ('eligible_differing_raw', 'all_clip_identical', 'forced_loss_present',
                     'SAFE2', 'opposite_signs', 'inside_and_outside', 'old_cap_active'):
            if a[flag]:
                keys.append(flag)
        for key in keys:
            counts[key] += 1
            by_class[cls][key] += 1
            games[key].add(case['provenance']['game_id'])
        for key in ('clip_masked_pairs', 'final_cap_masked_pairs', 'old_floor_masked_pairs'):
            counts[key] += a[key]
            by_class[cls][key] += a[key]
        for source in a['extreme_sources_compatible']:
            counts[source] += 1
        gaps[cls].append(a['raw_gap'])
        if a['eligible_differing_raw']:
            differing_gaps[cls].append(a['raw_gap'])
        spreads[cls].append(a['raw_spread'])
        if a['all_clip_identical']:
            d = case['diagnostic']
            lead = d['scores'][d['side']] - d['scores']['B' if d['side'] == 'A' else 'A']
            lead_key = 'allclip_ahead' if lead > 0 else 'allclip_behind' if lead < 0 else 'allclip_tied'
            counts[lead_key] += 1
            by_class[cls][lead_key] += 1
            games[cls + '/all_clip_identical'].add(case['provenance']['game_id'])
            key = cls + '/' + a['geometry']
            examples[key].append({'provenance': case['provenance'], 'analysis': a,
                                  'historic_direction': case['diagnostic']['direction']})
    top = []
    for key, rows in sorted(examples.items()):
        top.extend(sorted(rows, key=lambda r: r['analysis']['raw_gap'], reverse=True)[:3])
    return {'counts': dict(counts), 'by_terminal_class': {k: dict(v) for k, v in by_class.items()},
            'observed_terminal_categories_all_decisions': dict(observed_terminal_categories),
            'games_with_at_least_one': {k: len(v) for k, v in games.items()},
            'raw_top1_top2_gap': {k: distribution(v) for k, v in gaps.items()},
            'raw_top1_top2_gap_differing_raw': {k: distribution(v) for k, v in differing_gaps.items()},
            'raw_spread': {k: distribution(v) for k, v in spreads.items()},
            'top_examples': top,
            'candidate_formula_status': 'not approved; no hypothetical candidate computed'}, analyzed


def select_fixtures(analyzed, minimum=24):
    chosen, seen, categories = [], set(), set()
    def take(case, analysis):
        identity = (case['provenance']['filename'], case['provenance']['debug_line'])
        if identity not in seen:
            seen.add(identity)
            chosen.append(dict(case, analysis=analysis))
    for case, a in analyzed:
        d = case['diagnostic']
        special = (case['provenance']['game_id'].startswith('8d25279e') and case['provenance']['debug_line'] == 498
                   or case['provenance']['game_id'].startswith('5580bbd6') and case['provenance']['debug_line'] == 576)
        category = (a['terminal_class'], a['geometry'], a['old_cap_active'], a['forced_loss_present'],
                    tuple(a['extreme_sources_compatible']), a['SAFE2'], a['opposite_signs'])
        if special or (d.get('deep_moves') and category not in categories):
            categories.add(category)
            take(case, a)
    for case, a in sorted(analyzed, key=lambda pair: pair[1]['raw_gap'] or 0, reverse=True):
        if len(chosen) >= minimum:
            break
        if a['eligible_differing_raw']:
            take(case, a)
    return chosen


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + '\n')


def markdown(summary):
    lines = ['# Deep clipping phase 1: frozen historical corpus', '',
             'No candidate normalization, production edits, timed searches or games.', '',
             'Denominators: depth>=4 and >=2 non-forced raw scores includes raw ties; differing raw is a separate subset.',
             'Terminal-horizon: 0<=R<=completed depth. Extreme: abs(any raw)>=300M; compatible, not proof of terminal origin.',
             'Source returns +/-500M plus score delta for elimination, +/-300M plus delta for final score; zero can mean a terminal draw.', '',
             '| Metric | Count |', '|---|---:|']
    lines.extend(f'| {k} | {v} |' for k, v in summary['counts'].items())
    lines += ['', '## By Terminal Class', '', '```json', json.dumps(summary['by_terminal_class'], indent=2), '```',
              '', '## Real Observed Terminal Categories', '', '```json', json.dumps(summary['observed_terminal_categories_all_decisions'], indent=2), '```',
              '', 'Absent categories are not fabricated. Marker families are magnitude-compatible, not traced provenance.',
              '', '## Top1/Top2 Raw Gaps', '', 'Quantiles use lower-index empirical order statistics; median is the usual midpoint.',
              '', '```json', json.dumps(summary['raw_top1_top2_gap'], indent=2), '```',
              '', '## Games With At Least One', '', '```json', json.dumps(summary['games_with_at_least_one'], indent=2), '```',
              '', '## Examples', '']
    for row in summary['top_examples']:
        p, a = row['provenance'], row['analysis']
        lines.append(f"- {p['filename']}:{p['debug_line']} {a['terminal_class']}, raw {a['raw_top1']}/{a['raw_top2']}, gap {a['raw_gap']}, historic move {row['historic_direction']}")
    return '\n'.join(lines) + '\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('analyze', 'compact', 'replay', 'hypothesis', 'details', 'compare'))
    parser.add_argument('--report', type=Path)
    parser.add_argument('--frozen-manifest', type=Path)
    parser.add_argument('--games', type=Path, default=ROOT / 'games')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--baseline-search', type=Path)
    parser.add_argument('--candidate-search', type=Path)
    parser.add_argument('--candidate-sha256')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args(argv)
    if args.command == 'compact':
        fixture = json.loads(args.fixtures.read_text())
        histories = {c['provenance']['game_id'] + ':' + str(c['provenance']['debug_line']): c['history'] for c in fixture['cases']}
        if any(isinstance(c['history'], list) for c in fixture['cases']):
            with gzip.open(args.output / 'fixture_histories.json.gz', 'xb') as stream:
                stream.write(json.dumps(histories, sort_keys=True).encode())
        fixture['cases'] = [compact_fixture_history(c) for c in fixture['cases']]
        temporary = args.fixtures.with_suffix('.compact.json')
        write_new(temporary, json.dumps(fixture, sort_keys=True, separators=(',', ':')) + '\n')
        temporary.replace(args.fixtures)
        print(json.dumps({'fixture_count': len(fixture['cases']), 'bytes': args.fixtures.stat().st_size}))
        return
    if args.command == 'replay':
        source = load_baseline(args.baseline_search)
        fixture = json.loads(args.fixtures.read_text())
        rows = [replay_current_reference(c, source) for c in fixture['cases']]
        counts = Counter(r['status'] for r in rows)
        for flag in ('historic_vs_current_x_final_changed', 'current_x_plan_changed',
                     'historic_vs_current_x_adjustment_changed', 'historic_vs_current_x_order_changed', 'safe_repair'):
            counts[flag] = sum(bool(r.get(flag)) for r in rows)
        for flag in ('raw_result_preserved', 'shared_request_and_result_exact', 'hunger_exact', 'repeat_count_exact'):
            counts[flag] = sum(bool(r.get(flag)) for r in rows)
        game_counts = {flag: len({r['provenance']['game_id'] for r in rows if r.get(flag)})
                       for flag in ('historic_vs_current_x_final_changed', 'historic_vs_current_x_order_changed')}
        result = {'counts': dict(counts), 'pool': 'selected real fixtures; not representative corpus sample',
                  'games_with_at_least_one': game_counts, 'historical_request_certified_count': 0, 'results': rows}
        write_new(args.output / 'reference_replay.json', result)
        write_new(args.output / 'reference_replay.md', '# Current X64 frozen-score reference replay\n\n```json\n' + json.dumps(dict(counts), indent=2) + '\n```\n\nNo candidate brain executed. Changed-target cases excluded. Historic source/request not certified.\n')
        print(json.dumps(dict(counts), indent=2))
        return
    if args.report is None or args.frozen_manifest is None:
        parser.error('analyze requires --report and --frozen-manifest')
    cases = load_cohort(args.report, args.frozen_manifest, args.games)
    if args.command == 'compare':
        baseline = load_baseline(args.baseline_search)
        candidate = load_source(args.candidate_search, args.candidate_sha256)
        result = run_comparison(cases, baseline, candidate, args.output, args.limit)
        print(json.dumps(result['counts'], indent=2))
        return
    frozen = json.loads(args.frozen_manifest.read_text())
    names = manifest_names(args.report)
    if args.command == 'hypothesis':
        source = load_baseline(args.baseline_search)
        result = measure_hypothesis(cases, source)
        write_new(args.output / 'hypothesis.json', result)
        write_new(args.output / 'hypothesis.md', '# Approved formula: hypothetical historical ranking impact\n\n' + result['scope'] + '\n\n' + result['formula'] + '\n\n```json\n' + json.dumps({k: v for k, v in result.items() if k not in ('top_examples', 'cap_examples')}, indent=2) + '\n```\n\nUnchanged rankings are not attributed to cap except for the explicit b control. Full example rows in hypothesis.json.\n')
        print(json.dumps({k: v for k, v in result.items() if k not in ('top_examples', 'cap_examples')}, indent=2))
        return
    summary, analyzed = summarize(cases)
    if args.command == 'details':
        write_new(args.output / 'corpus_details.json', summary)
        write_new(args.output / 'corpus_details.md', markdown(summary))
        print(json.dumps({'counts': summary['counts'], 'terminal_categories': summary['observed_terminal_categories_all_decisions'],
                          'gaps_differing_raw': summary['raw_top1_top2_gap_differing_raw']}, indent=2))
        return
    summary['manifest'] = {'report_sha256': frozen['report_sha256'], 'filenames': names,
                           'log_hashes': frozen['log_hashes'], 'historical_declared_version': '2.16',
                           'historical_planner': 'pre-X fix; not cryptographically certified',
                           'current_reference_x_sha256': REFERENCE_X,
                           'move_search_baseline_sha256': REFERENCE_SEARCH}
    fixtures = select_fixtures(analyzed)
    summary['fixture_count'] = len(fixtures)
    summary['fixture_categories'] = dict(Counter(f['analysis']['terminal_class'] + '/' + f['analysis']['geometry'] for f in fixtures))
    histories = {c['provenance']['game_id'] + ':' + str(c['provenance']['debug_line']): c['history'] for c in fixtures}
    with gzip.open(args.output / 'fixture_histories.json.gz', 'xb') as stream:
        stream.write(json.dumps(histories, sort_keys=True).encode())
    compacted = [compact_fixture_history(c) for c in fixtures]
    write_new(args.fixtures, json.dumps({'manifest': summary['manifest'], 'cases': compacted}, sort_keys=True, separators=(',', ':')) + '\n')
    write_new(args.output / 'corpus.json', summary)
    write_new(args.output / 'corpus.md', markdown(summary))
    print(json.dumps({'counts': summary['counts'], 'by_terminal_class': summary['by_terminal_class'],
                      'raw_gaps': summary['raw_top1_top2_gap'], 'fixtures': summary['fixture_categories']}, indent=2))


if __name__ == '__main__':
    main()
