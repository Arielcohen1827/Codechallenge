"""Frozen real-log normalization audit. Never runs games or timed searches."""

from __future__ import annotations

import argparse
import atexit
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
import tempfile
from time import perf_counter
from types import SimpleNamespace, ModuleType
import zipfile

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_X = '64dd15b53bc755af443cee49948c368ce41bac3097a3eab32c5c95a6875e68e3'
REFERENCE_SEARCH = '09142bcc462f3ae8954e6527c21cb4f80784e1d0c4db2b10372db438a44d3dc4'
FROZEN_SOURCES = ROOT / 'tests/fixtures/deep_v7_sources.zip'
FROZEN_SUPPORT = ROOT / 'tests/fixtures/deep_v7_support.zip'
_support_directory = None
OBSERVED_FIELDS = (
    'game_id',
    'board',
    'side',
    'remaining_moves',
    'rows',
    'cols',
    'board_size',
    'score_1',
    'score_2',
    'multiplier_1',
    'multiplier_2',
)
CONTEXT_FIELDS = (
    'side',
    'head',
    'enemy_head',
    'scores',
    'multipliers',
    'food',
    'numbered_food',
    'next_food_digit',
    'pickups',
    'walls',
    'crash_history',
    'tail_food_a',
    'tail_food_b',
    'remaining_moves',
    'current_target',
    'hunger',
    'repeat_count',
    'plan',
    'direction',
    'reason',
    'legal',
    'active_games',
    'search_time_budget_ms',
    'commit_to_plan',
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def historical_support():
    """Materialize certified audit inputs outside the source/coverage tree."""
    global _support_directory
    if _support_directory is None:
        directory = tempfile.TemporaryDirectory(prefix='snake-v7-support-')
        root = Path(directory.name)
        try:
            with zipfile.ZipFile(FROZEN_SUPPORT) as archive:
                manifest = json.loads(archive.read('provenance.json'))
                for name, expected in manifest['files'].items():
                    path = root / name
                    path.resolve().relative_to(root.resolve())
                    raw = archive.read(name)
                    if digest(raw) != expected:
                        raise ValueError('Historical support hash mismatch: ' + name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_bytes(raw)
                (root / 'provenance.json').write_bytes(archive.read('provenance.json'))
        except BaseException:
            directory.cleanup()
            raise
        _support_directory = directory
        atexit.register(directory.cleanup)
    return Path(_support_directory.name)


def normalized(value):
    return json.loads(json.dumps(value, sort_keys=True))


def manifest_names(report):
    section = (
        Path(report).read_text(encoding='utf-8').split('## Manifiesto de la tanda fija', 1)[1]
    )
    names = sorted(set(re.findall(r'game_[a-f0-9-]+\.log', section)))
    if len(names) != 86:
        raise ValueError('Expected frozen 86-file report manifest')
    return names


def load_cohort(report, frozen_manifest, games):
    frozen = json.loads(Path(frozen_manifest).read_text())
    names = manifest_names(report)
    if (
        names != sorted(frozen['filenames'])
        or digest(Path(report).read_bytes()) != frozen['report_sha256']
    ):
        raise ValueError('Frozen report/manifest mismatch')
    cases = []
    for name in names:
        path = Path(games) / name
        if digest(path.read_bytes()) != frozen['log_hashes'][name]:
            raise ValueError('Frozen log hash mismatch: ' + name)
        cases.extend(read_log(path))
    return cases


def read_log(path):
    """Decode supported historical records with their exact source provenance."""
    path = Path(path)
    raw = path.read_bytes()
    reader = HistoricalLogReader(path.name, digest(raw))
    for number, line in enumerate(raw.decode('utf-8-sig').splitlines(), 1):
        if line.startswith(('= ', '< ', '? ')):
            reader.record(json.loads(line[2:]), number, line)
    return reader.cases


class HistoricalLogReader:
    """Track the current observed turn independently from decision-debug history."""

    def __init__(self, filename, log_hash):
        self.filename, self.log_hash = filename, log_hash
        self.observed = self.observed_line = self.version = None
        self.history, self.cases = [], []

    def record(self, event, number, line):
        kind = event.get('event')
        if kind == 'bot_version':
            self.version = event.get('version')
        elif kind == 'your_turn':
            self.observed = observed_fields(event['data'])
            self.observed_line = number
        elif kind == 'decision_debug':
            self.decision(event, number, line)

    def decision(self, event, number, line):
        context = decision_context(event)
        case = {
            'provenance': self.provenance(event, number, line),
            'observed': self.observed,
            'diagnostic': decision_diagnostic(event, context),
            'history': list(self.history[-80:]),
        }
        self.cases.append(case)
        self.history.append(
            {
                'observed': self.observed,
                'observed_line': self.observed_line,
                'debug_line': number,
                'context': context,
            }
        )

    def provenance(self, event, number, line):
        return {
            'filename': self.filename,
            'game_id': event['game_id'],
            'debug_line': number,
            'observed_line': self.observed_line,
            'log_sha256': self.log_hash,
            'debug_sha256': digest(line.encode()),
            'declared_version': self.version,
            'source': 'real_server_log',
        }


def observed_fields(data):
    return {key: data[key] for key in OBSERVED_FIELDS if key in data}


def decision_context(event):
    snapshot = event.get('verification', {}).get('after') or {}
    context = {key: event[key] for key in CONTEXT_FIELDS if key in event}
    context['tracked_body'] = snapshot.get('tracked_body', {})
    return context


def decision_diagnostic(event, context):
    diagnostic = dict(context)
    for field in ('deep_moves', 'food_options', 'candidate_moves', 'commit_to_plan'):
        if field in event:
            diagnostic[field] = event[field]
    return diagnostic


def terminal_class(values, remaining, depth):
    if remaining is not None and 0 <= remaining <= depth:
        return 'terminal-horizon'
    if any(abs(v) >= 300_000_000 for v in values):
        return 'terminal-compatible/extreme'
    return 'normal'


def legacy_adjustments(rows, depth, percent=35, cap=25000, minimum_depth=4):
    result = {r['direction']: 0 for r in rows}
    usable = usable_score_rows(rows)
    if depth < minimum_depth or not usable:
        return result
    result.update(legacy_centered_values(usable, percent, cap))
    return result


def legacy_centered_values(usable, percent, cap):
    clipped = {row['direction']: max(-200000, min(200000, row['search_score'])) for row in usable}
    center = sum(clipped.values()) // len(clipped)
    return {
        key: max(-cap, min(cap, (value - center) * percent // 100))
        for key, value in clipped.items()
    }


def hypothetical_adjustments(rows, depth, remaining, root_terminal=False):
    """Approved formula for offline measurement only; never installed in a brain."""
    usable = usable_score_rows(rows)
    if hypothetical_score_guard(usable, depth, remaining, root_terminal):
        return legacy_adjustments(rows, depth)
    return hypothetical_centered_values(rows, usable)


def hypothetical_score_guard(usable, depth, remaining, root_terminal):
    guarded = root_terminal or horizon_terminal(remaining, depth)
    guarded |= any(abs(row['search_score']) >= 300000000 for row in usable)
    return guarded or not hypothetical_depth_eligible(depth, usable)


def horizon_terminal(remaining, depth):
    return remaining is not None and remaining <= depth


def hypothetical_depth_eligible(depth, usable):
    return depth >= 4 and bool(usable)


def hypothetical_centered_values(rows, usable):
    center = sum(r['search_score'] for r in usable) // len(usable)
    result = {r['direction']: 0 for r in rows}
    result.update(
        {
            r['direction']: max(-25000, min(25000, (r['search_score'] - center) * 35 // 100))
            for r in usable
        }
    )
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
    rows = unadjusted_recorded_rows(d, source)
    depth = max((r.get('search_depth', 0) for r in rows), default=0)
    root_terminal = recorded_terminal(d)
    adjustments = (
        hypothetical_adjustments(rows, depth, d.get('remaining_moves'), root_terminal)
        if candidate
        else legacy_adjustments(rows, depth)
    )
    if override is not None:
        adjustments = override
    apply_recorded_adjustments(rows, adjustments)
    rows.sort(key=source.deep_sort_key, reverse=True)
    source.prefer_safer_close_move(rows)
    return rows


def unadjusted_recorded_rows(diagnostic, source):
    order = {direction: index for index, direction in enumerate(diagnostic.get('legal', []))}
    rows = sorted(
        copy.deepcopy(diagnostic.get('deep_moves', [])),
        key=lambda row: order.get(row['direction'], 99),
    )
    for row in rows:
        row['score'] -= row.get('search_adjustment', 0)
    rows.sort(key=source.deep_sort_key, reverse=True)
    return rows


def apply_recorded_adjustments(rows, adjustments):
    for row in rows:
        row['search_adjustment'] = adjustments[row['direction']]
        row['score'] += adjustments[row['direction']]


def recorded_terminal(diagnostic):
    return diagnostic.get('remaining_moves', 1) <= 0 or any(
        value < -2500 for value in diagnostic['scores'].values()
    )


def measure_hypothesis(cases, source):
    """Summarize existing recorded cases with the historical measurement schema."""
    counts, games, gaps, examples = Counter(), defaultdict(set), defaultdict(list), []
    for case in cases:
        analysis = analyze_case(case)
        rows = case['diagnostic'].get('deep_moves', [])
        old_rank = rerank_recorded(case, source)
        counts['legacy_order_control_mismatch'] += old_rank != rows
        if not analysis['eligible_depth_and_two']:
            continue
        counts['eligible_depth_and_two'] += 1
        hypothesis = recorded_hypothesis(case, analysis, old_rank)
        collect_hypothesis(case, analysis, hypothesis, counts, games, gaps, examples, source)
    return hypothesis_summary(counts, games, gaps, examples)


def recorded_hypothesis(case, analysis, old_rank):
    diagnostic = case['diagnostic']
    rows = diagnostic.get('deep_moves', [])
    usable = sorted(usable_score_rows(rows), key=lambda row: row['search_score'], reverse=True)
    terminal = recorded_terminal(diagnostic)
    guarded = terminal or diagnostic['remaining_moves'] <= analysis['depth']
    guarded |= any(abs(row['search_score']) >= 300000000 for row in usable)
    return {
        'guarded': guarded,
        'usable': usable,
        'old_rank': old_rank,
        'old': legacy_adjustments(rows, analysis['depth']),
        'capped': hypothetical_adjustments(
            rows, analysis['depth'], diagnostic['remaining_moves'], terminal
        ),
    }


def collect_hypothesis(case, analysis, hypothesis, counts, games, gaps, examples, source):
    if hypothesis['guarded']:
        counts['guarded_legacy_whole_group'] += 1
        counts['guarded_mapping_changed'] += hypothesis['capped'] != hypothesis['old']
        return
    counts['normal_eligible'] += 1
    counts['normal_differing_raw'] += analysis['eligible_differing_raw']
    record = normal_hypothesis_record(case, analysis, hypothesis, source)
    collect_normal_hypothesis(case, analysis, record, counts, games, gaps, examples)


def uncapped_hypothesis_values(rows, usable):
    center = sum(row['search_score'] for row in usable) // len(usable)
    uncapped = {row['direction']: (row['search_score'] - center) * 35 // 100 for row in usable}
    uncapped.update({row['direction']: 0 for row in rows if row['direction'] not in uncapped})
    return uncapped


def ranked_record_directions(rows):
    return [row['direction'] for row in rows]


def normal_hypothesis_record(case, analysis, hypothesis, source):
    rows = case['diagnostic'].get('deep_moves', [])
    uncapped = uncapped_hypothesis_values(rows, hypothesis['usable'])
    orders = {
        'old': ranked_record_directions(hypothesis['old_rank']),
        'new': ranked_record_directions(rerank_recorded(case, source, candidate=True)),
        'uncapped': ranked_record_directions(rerank_recorded(case, source, override=uncapped)),
    }
    first, second = hypothesis['usable'][:2]
    uncapped_pair = (uncapped[first['direction']], uncapped[second['direction']])
    capped_pair = (
        hypothesis['capped'][first['direction']],
        hypothesis['capped'][second['direction']],
    )
    flags = hypothesis_flags(analysis, hypothesis, uncapped, orders)
    flags.update(hypothesis_cap_flags(hypothesis, uncapped, orders, uncapped_pair, capped_pair))
    return {
        'provenance': case['provenance'],
        'raw_gap': analysis['raw_gap'],
        'raw_top1': analysis['raw_top1'],
        'raw_top2': analysis['raw_top2'],
        'old_adjustments': hypothesis['old'],
        'uncapped': uncapped,
        'capped': hypothesis['capped'],
        'old_order': orders['old'],
        'uncapped_order': orders['uncapped'],
        'capped_order': orders['new'],
        'flags': flags,
        'uncapped_pair': uncapped_pair,
        'capped_pair': capped_pair,
    }


def hypothesis_flags(analysis, hypothesis, uncapped, orders):
    changed = hypothesis['capped'] != hypothesis['old']
    order_changed = orders['new'] != orders['old']
    return {
        'adjustment_changed': changed,
        'order_changed': order_changed,
        'top_ranked_move_changed': orders['new'][0] != orders['old'][0],
        'uncapped_order_changed': orders['uncapped'] != orders['old'],
        'uncapped_mapping_changed': uncapped != hypothesis['old'],
        'normal_old_flat_loss': analysis['all_clip_identical'],
        'normal_old_flat_recovered_adjustment': analysis['all_clip_identical'] and changed,
        'normal_old_flat_recovered_order': analysis['all_clip_identical'] and order_changed,
        'new_final_cap_active': any(abs(value) > 25000 for value in uncapped.values()),
    }


def hypothesis_cap_flags(hypothesis, uncapped, orders, uncapped_pair, capped_pair):
    return {
        'cap_a_top1_top2_gap_lost': uncapped_pair[0] != uncapped_pair[1]
        and capped_pair[0] == capped_pair[1],
        'cap_b_uncapped_order_changes_but_capped_stays_old': orders['uncapped'] != orders['old']
        and orders['new'] == orders['old'],
        'cap_c_uncapped_mapping_changes_but_capped_stays_old': uncapped != hypothesis['old']
        and hypothesis['capped'] == hypothesis['old'],
    }


def collect_normal_hypothesis(case, analysis, record, counts, games, gaps, examples):
    first, second = record['uncapped_pair']
    c_first, c_second = record['capped_pair']
    counts['raw_top1_top2_positive_gap'] += analysis['raw_gap'] > 0
    counts['uncapped_top1_top2_positive_gap'] += first != second
    for key, value in record['flags'].items():
        counts[key] += value
        if value:
            games[key].add(case['provenance']['game_id'])
    gaps['normal_raw_top1_top2'].append(analysis['raw_gap'])
    gaps['normal_uncapped_top1_top2'].append(first - second)
    gaps['normal_capped_top1_top2'].append(c_first - c_second)
    collect_hypothesis_example(analysis, record, gaps, examples)


def has_cap_flag(record):
    return any(record['flags'][key] for key in record['flags'] if key.startswith('cap_'))


def collect_hypothesis_example(analysis, record, gaps, examples):
    if analysis['eligible_differing_raw']:
        gaps['normal_differing_raw_top1_top2'].append(analysis['raw_gap'])
    if record['flags']['order_changed'] or has_cap_flag(record):
        examples.append(public_hypothesis_example(record))


def public_hypothesis_example(record):
    return {
        key: value for key, value in record.items() if key not in ('uncapped_pair', 'capped_pair')
    }


def hypothesis_summary(counts, games, gaps, examples):

    denominator = counts['normal_eligible']
    return {
        'formula': (
            'applicable non-forced raw integer mean -> 35% floor -> +/-25000 '
            'cap; whole old group if applicable extreme >=300M, R<=depth or '
            'root terminal'
        ),
        'scope': (
            'hypothetical frozen historical shallow bases; no candidate brain '
            'or competitive benefit claim'
        ),
        'counts': dict(counts),
        'games_with_at_least_one': {k: len(v) for k, v in games.items()},
        'cap_definitions_and_denominators': {
            'a': {
                'definition': (
                    'uncapped top1/top2 adjustments differ but capped ' 'adjustments equal'
                ),
                'normal_pool': denominator,
                'uncapped_positive_top1_top2_gap': counts['uncapped_top1_top2_positive_gap'],
            },
            'b': {
                'definition': (
                    'uncapped full order differs from old; capped full order '
                    'equals old, after unchanged sort/prefer'
                ),
                'normal_pool': denominator,
                'uncapped_order_changed': counts['uncapped_order_changed'],
            },
            'c': {
                'definition': (
                    'uncapped adjustment map differs from old; capped map ' 'equals old'
                ),
                'normal_pool': denominator,
                'uncapped_mapping_changed': counts['uncapped_mapping_changed'],
            },
        },
        'gap_distributions': {k: distribution(v) for k, v in gaps.items()},
        'top_examples': sorted(examples, key=lambda r: r['raw_gap'], reverse=True)[:12],
        'cap_examples': [record for record in examples if has_cap_flag(record)][:8],
    }


def state_for(case):
    from snake_bot.snake_state import parse_state

    d = case['diagnostic']
    state = parse_state(case['observed'], crash_history=d.get('crash_history'))
    state = restore_recorded_bodies(state, d['tracked_body'])
    verify_recorded_state(state, d)
    verify_recorded_moves(state, d)
    return state, {
        'observable_state_exact': True,
        'tracked_body_order_from_log': True,
        'history_counters_from_log': True,
        'unlogged_internal_cache_history_certified': False,
    }


def restore_recorded_bodies(state, tracked):
    if set(tracked) != {'A', 'B'}:
        raise ValueError('Missing real tracked bodies')
    bodies = {side: tuple(map(tuple, tracked[side]['body'])) for side in ('A', 'B')}
    if any(set(bodies[side]) != set(state.body(side)) for side in ('A', 'B')):
        raise ValueError('Tracked body cells disagree with board')
    return replace(
        state,
        snakes=bodies,
        reliable_tails=recorded_reliable_tails(tracked),
    )


def recorded_reliable_tails(tracked):
    return frozenset(side for side in ('A', 'B') if tracked[side]['order_reliable'])


def verify_recorded_state(state, d):
    fields = {
        'side': state.side,
        'head': state.head(state.side),
        'enemy_head': state.head(state.enemy),
        'remaining_moves': state.remaining_moves,
        'scores': state.scores,
        'multipliers': state.multipliers,
        'food': sorted(state.food),
        'numbered_food': sorted(state.food_values.items()),
        'next_food_digit': state.next_food_digit,
        'pickups': sorted(state.pickups),
        'walls': sorted(state.walls),
        'tail_food_a': sorted(state.tail_food_a),
        'tail_food_b': sorted(state.tail_food_b),
    }
    for name, value in fields.items():
        if normalized(value) != normalized(d[name]):
            raise ValueError('Recorded state mismatch: ' + name)


def verify_recorded_moves(state, d):
    from snake_bot.snake_state import legal_moves

    legal = legal_moves(state, state.side)
    clean = [m for m in legal if clean_recorded_move(state, m)]
    if set(clean or legal) != set(d['legal']):
        raise ValueError('Recorded legal moves mismatch')


def clean_recorded_move(state, direction):
    from snake_bot.snake_state import step

    target = step(state.head(state.side), direction)
    return target not in state.wrong_food() and target not in state.walls


def compact_fixture_history(case, signature_cache=None):
    """Compact recorded history using unchanged target assignment and canonical signatures."""
    if isinstance(case['history'], dict):
        return normalize_compact_history(case)
    builder = RecordedHistoryCompactor(case, signature_cache)
    for entry in case['history']:
        builder.record(entry)
    return builder.compacted_case()


def normalize_compact_history(case):
    compacted = copy.deepcopy(case)
    if compacted['history'].get('signature_format') != 'sha256-canonical-json':
        compacted['history']['signatures'] = [
            canonical_signature(value) for value in compacted['history']['signatures']
        ]
        compacted['history']['signature_format'] = 'sha256-canonical-json'
    return compacted


def canonical_signature(value):
    return digest(json.dumps(value, sort_keys=True, separators=(',', ':')).encode())


class RecordedHistoryCompactor:
    """Reconstruct historical remembered signatures without rerunning timed search."""

    def __init__(self, case, signature_cache):
        from snake_bot.snake_brain import SnakeBrain

        self.case, self.cache = case, signature_cache
        self.brain, self.lines = SnakeBrain(), None
        self.game_id = case['provenance']['game_id']
        self.provenance, self.signatures = [], []

    def record(self, entry):
        identity = (self.game_id, entry['debug_line'])
        if self.cache is not None and identity in self.cache:
            signature, trace = self.cache[identity]
            self.signatures.append(signature)
            self.provenance.append(trace)
            return
        context = entry['context']
        if not usable_history_context(context):
            return
        state, _ = state_for({'observed': entry['observed'], 'diagnostic': context})
        target = recorded_history_target(context, self.recorded_decision(entry))
        self.brain.targets[(self.game_id, state.side)] = target
        self.brain._remember(self.game_id, state)
        self.remember_signature(entry, identity, target)

    def recorded_decision(self, entry):
        context = entry['context']
        if 'commit_to_plan' in context:
            return context
        if self.lines is None:
            self.lines = (
                (ROOT / 'games' / self.case['provenance']['filename'])
                .read_text(encoding='utf-8-sig')
                .splitlines()
            )
        return json.loads(self.lines[entry['debug_line'] - 1][2:])

    def remember_signature(self, entry, identity, target):
        signature = canonical_signature(normalized(self.brain.recent_states[self.game_id][-1]))
        trace = {
            'debug_line': entry['debug_line'],
            'observed_line': entry['observed_line'],
            'post_target': target,
            'derivation': 'recorded branch plus unchanged target assignment',
        }
        self.signatures.append(signature)
        self.provenance.append(trace)
        if self.cache is not None:
            self.cache[identity] = (signature, trace)

    def compacted_case(self):
        compacted = dict(self.case)
        compacted['history'] = {
            'signatures': self.signatures,
            'signature_format': 'sha256-canonical-json',
            'signature_provenance': self.provenance,
            'full_history_sha256': digest(
                json.dumps(self.case['history'], sort_keys=True).encode()
            ),
            'external_history_key': self.game_id
            + ':'
            + str(self.case['provenance']['debug_line']),
        }
        return compact_fixture_history(compacted)


def usable_history_context(context):
    return bool(context.get('tracked_body')) and (
        bool(context.get('current_target')) or 'hunger' in context
    )


def recorded_history_target(context, recorded):
    from snake_bot.food_planner import should_use_plan_for_positioning

    plan = context.get('plan')
    if context['reason'] == 'advantage_control' or not plan:
        return None
    if committed_history_plan(plan, context, recorded):
        return tuple(plan['food'])
    if should_use_plan_for_positioning(SimpleNamespace(**plan)):
        return tuple(plan['food'])
    return None


def committed_history_plan(plan, context, recorded):
    return recorded.get('commit_to_plan') and plan['first_move'] in context['legal']


def hydrate_brain(case, brain):
    from collections import deque

    state, verification = state_for(case)
    d = case['diagnostic']
    game_id, side = case['provenance']['game_id'], state.side
    key = (game_id, side)
    brain.targets[key] = (
        tuple(d['current_target']) if d.get('current_target') is not None else None
    )
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
        raise ValueError(
            (f'Recorded repeat history mismatch: {repeat} != ' f'{d.get("repeat_count")}')
        )
    return state, verification


def hashed_repeat_count(self, game_id, state):
    from snake_bot.snake_brain import SnakeBrain

    probe = SnakeBrain()
    probe.targets[(game_id, state.side)] = self.targets.get((game_id, state.side))
    probe._remember(game_id, state)
    signature = digest(
        json.dumps(
            normalized(probe.recent_states[game_id][0]),
            sort_keys=True,
            separators=(',', ':'),
        ).encode()
    )
    return sum(value == signature for value in self.recent_states.get(game_id, ()))


def replay_current_reference(case, source, candidate=None):
    """Replay supplied historical search results without launching a timed search."""
    replay = HistoricalReferenceReplay(case)
    versions = (source, source if candidate is None else candidate)
    for ranking_source in versions:
        failure = replay.version(ranking_source)
        if failure is not None:
            return failure
    return replay.result(candidate is not None)


class HistoricalReferenceReplay:
    """Keep common replay inputs, frozen plans and outcome provenance together."""

    def __init__(self, case):
        from snake_bot import bot_weights

        self.case, self.diagnostic = case, case['diagnostic']
        self.weights = dict(bot_weights.get_weights())
        self.planner = load_reference_planner(self.weights)
        self.tape = FrozenEngine(case)
        self.results, self.frozen_plans = [], None
        self.verification = None
        self.record = {
            'provenance': case['provenance'],
            'historic_direction': self.diagnostic['direction'],
            'historic_plan': self.diagnostic.get('plan'),
            'comparison': 'current X64 legacy baseline versus historic 2.16 pre-X log',
        }

    def version(self, ranking_source):
        brain, namespace = self.brain(ranking_source)
        try:
            state, self.verification = hydrate_brain(self.case, brain)
        except ValueError as error:
            return dict(self.record, status='unsupported_context', exclusion=str(error))
        hunger = self.diagnostic.get('hunger', 0)
        target = brain.targets[(self.case['provenance']['game_id'], state.side)]
        if self.frozen_plans is None:
            self.frozen_plans = self.planner.build_food_plans(state, state.side, target, hunger)
        plan = self.prepare_plan(brain, namespace, state, target, hunger)
        if not recorded_target_matches(plan, self.diagnostic):
            return dict(
                self.record,
                status='unsupported_changed_target',
                exclusion=(
                    'old root score target differs from current X64 target; '
                    'no retimed search authorized'
                ),
            )
        return self.choose(brain)

    def brain(self, ranking_source):
        from snake_bot import snake_brain
        from tools.compare_food_planners import private_namespace, private_class

        ranking = private_namespace(ranking_source, self.planner, self.weights)
        cache_class = private_class(ranking_source.TurnSearchCache, ranking)
        ranking['TurnSearchCache'] = cache_class
        ranking['IterativeSearchEngine'] = self.tape
        namespace = private_namespace(snake_brain, self.planner, self.weights)
        namespace['TurnSearchCache'] = cache_class
        namespace['rank_deep_moves'] = ranking['rank_deep_moves']
        brain_class = private_class(snake_brain.SnakeBrain, namespace)
        brain_class._repeat_count = hashed_repeat_count
        return brain_class(enable_debug=True), namespace

    def prepare_plan(self, brain, namespace, state, target, hunger):
        def cached_plans(actual_state, actual_side, actual_target, actual_hunger):
            if (
                actual_state != state
                or actual_side != state.side
                or actual_target != target
                or actual_hunger != hunger
            ):
                raise ValueError('Planner inputs differ between replay versions')
            return list(self.frozen_plans)

        namespace['build_food_plans'] = cached_plans
        plan = self.planner.choose_food_plan(
            state, state.side, target, hunger, plans=self.frozen_plans
        )
        self.record['current_x_plan'] = normalized(brain._plan_debug(plan))
        return plan

    def choose(self, brain):
        if not self.diagnostic.get('deep_moves'):
            return dict(
                self.record,
                status='unsupported_no_search_result',
                exclusion='no historical SearchResult in decision',
            )
        try:
            move = brain.choose_move(copy.deepcopy(self.case['observed']))
            final = brain.safe_direction(self.case['observed'], move)
        except ValueError as error:
            return dict(self.record, status='unsupported_search_request', exclusion=str(error))
        debug = brain.decision_debug(self.case['provenance']['game_id'])
        self.results.append(
            {'choose_direction': move, 'final_direction': final, 'debug': copy.deepcopy(debug)}
        )
        return None

    def result(self, with_candidate):
        first, control = self.results
        result = historical_result(
            self.record, self.diagnostic, first, control, self.verification, self.tape
        )
        if with_candidate:
            update_candidate_result(result, first, control, self.tape)
        return result


def recorded_target_matches(plan, diagnostic):
    target = None if plan is None else plan.food
    return normalized(target) == (diagnostic.get('plan') or {}).get('food')


def raw_score_map(rows):
    return {
        row['direction']: row.get('search_score')
        for row in rows
        if row.get('search_score') is not None
    }


def shallow_base_map(debug):
    result = {}
    for original in debug.get('deep_moves', []):
        row = dict(original)
        row['score'] -= row.pop('search_adjustment', 0)
        result[row['direction']] = row
    return result


def row_directions(rows):
    return [row['direction'] for row in rows]


def historical_adjustment_changed(current_rows, historical_rows):
    historical = {row['direction']: row for row in historical_rows}
    return any(
        row.get('search_adjustment')
        != historical.get(row['direction'], {}).get('search_adjustment')
        for row in current_rows
    )


def historical_result(record, diagnostic, first, control, verification, tape):
    historic_rows = diagnostic.get('deep_moves', [])
    current_rows = first['debug'].get('deep_moves', [])
    shallow_exact = shallow_base_map(first['debug']) == shallow_base_map(control['debug'])
    return dict(
        record,
        status='controlled_historical_raw_replay',
        verification=verification,
        request=tape.request,
        search_calls=tape.calls,
        raw_result_preserved=raw_score_map(current_rows) == tape.result.scores,
        shared_request_and_result_exact=shallow_exact and tape.calls == 2,
        shallow_bases_and_info_exact=shallow_exact,
        historical_request_certified=False,
        historical_request_unknowns=[
            'unlogged max_depth',
            'unlogged full compact state/target request',
            'unlogged process weights/source hashes',
        ],
        repeat_count_exact=first['debug'].get('repeat_count') == diagnostic.get('repeat_count'),
        hunger_exact=first['debug'].get('hunger') == diagnostic.get('hunger'),
        safe_direction_applied=True,
        safe_repair=first['choose_direction'] != first['final_direction'],
        current_x_final_direction=first['final_direction'],
        control_final_direction=control['final_direction'],
        historic_vs_current_x_final_changed=diagnostic['direction'] != first['final_direction'],
        current_x_plan_changed=normalized(diagnostic.get('plan'))
        != normalized(first['debug'].get('plan')),
        historic_vs_current_x_adjustment_changed=historical_adjustment_changed(
            current_rows, historic_rows
        ),
        historic_vs_current_x_order_changed=row_directions(historic_rows)
        != row_directions(current_rows),
        current_x_debug=first['debug'],
    )


def update_candidate_result(result, first, control, tape):
    current_rows = first['debug'].get('deep_moves', [])
    candidate_rows = control['debug'].get('deep_moves', [])
    current_map = {row['direction']: row for row in current_rows}
    result.update(
        candidate_final_direction=control['final_direction'],
        candidate_final_changed=control['final_direction'] != first['final_direction'],
        candidate_choose_changed=control['choose_direction'] != first['choose_direction'],
        candidate_safe_repair=control['choose_direction'] != control['final_direction'],
        candidate_adjustment_changed=any(
            row['search_adjustment'] != current_map[row['direction']]['search_adjustment']
            for row in candidate_rows
        ),
        candidate_order_changed=row_directions(current_rows) != row_directions(candidate_rows),
        candidate_top_ranked_changed=current_rows[0]['direction']
        != candidate_rows[0]['direction'],
        raw_result_preserved=result['raw_result_preserved']
        and raw_score_map(candidate_rows) == tape.result.scores,
        candidate_debug=control['debug'],
    )


def replay_comparison(case, baseline, candidate):
    return replay_current_reference(case, baseline, candidate)


def comparison_counts(records, eligible_total):
    counts = Counter(
        eligible_historical_pool=eligible_total,
        processed=len(records),
        supported_common_input=0,
        historical_request_certified=0,
    )
    games = defaultdict(set)
    flags = (
        'candidate_adjustment_changed',
        'candidate_order_changed',
        'candidate_top_ranked_changed',
        'candidate_final_changed',
        'candidate_choose_changed',
        'safe_repair',
        'candidate_safe_repair',
        'historic_vs_current_x_final_changed',
        'historic_vs_current_x_adjustment_changed',
        'historic_vs_current_x_order_changed',
        'raw_result_preserved',
        'shared_request_and_result_exact',
        'shallow_bases_and_info_exact',
        'repeat_count_exact',
        'hunger_exact',
    )
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
    return counts, games


def comparison_summary(records, eligible_total, manifest, guard_controls):
    """Report supported replays, exclusions and invariant counts from recorded results."""
    counts, games = comparison_counts(records, eligible_total)
    examples = changed_comparison_examples(records)
    exclusions = comparison_exclusions(records)
    return {
        'counts': dict(counts),
        'games_with_at_least_one': {k: len(v) for k, v in games.items()},
        'guard_controls': guard_controls,
        'timed_searches': 0,
        'manifest': manifest,
        'final_move_examples': examples,
        'exclusions': exclusions,
        'interpretation': (
            'Actual current-X64 brain before/after followed by '
            'safe_direction; historical raw target matches required; unlogged '
            'historical request/source not certified; one-step control, no '
            'game counterfactual or competitive benefit claim'
        ),
    }


def changed_comparison_examples(records):
    return [record for record in records if record.get('candidate_final_changed')][:15]


def comparison_exclusions(records):
    return [record for record in records if record['status'] != 'controlled_historical_raw_replay']


def run_comparison(cases, baseline, candidate, output, limit=None):
    """Replay selected recorded cases and preserve incremental evidence on disk."""
    return ComparisonRun(cases, baseline, candidate, output, limit).run()


class ComparisonRun:
    """Own one bounded historical comparison and its invariant checks."""

    def __init__(self, cases, baseline, candidate, output, limit):
        self.cases, self.baseline, self.candidate = cases, baseline, candidate
        self.output, self.limit = Path(output), limit
        self.records, self.signature_cache = [], {}
        self.guarded, self.guard_mismatches = 0, 0

    def run(self):
        """Persist each replay row before producing the aggregate comparison report."""
        self.output.mkdir(parents=True, exist_ok=True)
        self.eligible = eligible_comparison_cases(self.cases, self.limit)
        self.started = perf_counter()
        with gzip.open(self.output / 'comparison_rows.jsonl.gz', 'xt', encoding='utf-8') as stream:
            for index, case in enumerate(self.eligible, 1):
                self.inspect_guard(case)
                record = self.replay_case(case)
                stream.write(json.dumps(record, sort_keys=True, separators=(',', ':')) + '\n')
                verify_comparison_invariants(record)
                self.records.append(small_comparison_record(record))
                self.report_progress(index, stream)
        manifest = comparison_source_manifest(self.cases, self.baseline, self.candidate)
        result = comparison_summary(
            self.records,
            len(self.eligible),
            manifest,
            {
                'historical_guarded_cases': self.guarded,
                'actual_helper_legacy_mapping_mismatches': self.guard_mismatches,
            },
        )
        write_new(self.output / 'comparison.json', result)
        write_comparison_markdown(self.output, result)
        return result

    def inspect_guard(self, case):
        from snake_bot.bot_weights import get_weights

        analysis = analyze_case(case)
        if analysis['terminal_class'] == 'normal':
            return
        state, _ = state_for(case)
        self.guarded += 1
        actual = self.candidate._search_adjustments(
            state, case['diagnostic']['deep_moves'], analysis['depth'], get_weights()
        )
        old = legacy_adjustments(case['diagnostic']['deep_moves'], analysis['depth'])
        self.guard_mismatches += any(actual.get(key, 0) != value for key, value in old.items())

    def replay_case(self, case):
        try:
            compacted = compact_fixture_history(case, self.signature_cache)
            return replay_comparison(compacted, self.baseline, self.candidate)
        except ValueError as error:
            return {
                'provenance': case['provenance'],
                'status': 'unsupported_context',
                'exclusion': str(error),
            }

    def report_progress(self, index, stream):
        if index % 100 == 0 or index == len(self.eligible):
            stream.flush()
            supported = sum(
                record['status'] == 'controlled_historical_raw_replay' for record in self.records
            )
            changed = sum(bool(record.get('candidate_final_changed')) for record in self.records)
            elapsed = perf_counter() - self.started
            eta = elapsed / index * (len(self.eligible) - index)
            print(
                json.dumps(
                    {
                        'processed': index,
                        'total': len(self.eligible),
                        'supported': supported,
                        'final_changed': changed,
                        'elapsed_seconds': round(elapsed, 1),
                        'eta_seconds': round(eta, 1),
                    }
                ),
                flush=True,
            )


def eligible_comparison_cases(cases, limit):
    eligible = [case for case in cases if analyze_case(case)['eligible_depth_and_two']]
    return eligible if limit is None else eligible[:limit]


def verify_comparison_invariants(record):
    if record['status'] == 'controlled_historical_raw_replay' and not all(
        record[key]
        for key in (
            'raw_result_preserved',
            'shared_request_and_result_exact',
            'shallow_bases_and_info_exact',
            'hunger_exact',
            'repeat_count_exact',
        )
    ):
        raise ValueError('Replay invariant failed: ' + str(record['provenance']))


def small_comparison_record(record):
    small = {
        key: value
        for key, value in record.items()
        if key not in ('current_x_debug', 'candidate_debug', 'request')
    }
    if record.get('candidate_final_changed'):
        small['before_reason'] = record['current_x_debug']['reason']
        small['after_reason'] = record['candidate_debug']['reason']
    return small


def comparison_source_manifest(cases, baseline, candidate):
    from snake_bot.bot_weights import get_weights

    source_manifest = {
        'baseline_move_search_sha256': digest(Path(baseline.__file__).read_bytes()),
        'candidate_move_search_sha256': digest(Path(candidate.__file__).read_bytes()),
        'planner_x_sha256': REFERENCE_X,
        'engine_sha256': digest((ROOT / 'snake_bot/search_engine.py').read_bytes()),
        'weights_sha256': digest((ROOT / 'weights/active_weights.json').read_bytes()),
        'normalization_percent': get_weights()['iterative_search_weight_percent'],
        'normalization_cap': get_weights()['iterative_search_adjustment_cap'],
        'cohort_log_hashes': {
            c['provenance']['filename']: c['provenance']['log_sha256'] for c in cases
        },
    }
    return source_manifest


def write_comparison_markdown(output, result):
    text = (
        '# Current X64 exact common-input one-step replay\n\n'
        + result['interpretation']
        + '\n\n```json\n'
        + json.dumps(
            {k: v for k, v in result.items() if k not in ('exclusions', 'final_move_examples')},
            indent=2,
        )
        + '\n```\n\n## Final Move Examples\n\n'
    )
    for row in result['final_move_examples']:
        p = row['provenance']
        text += (
            f"- {p['filename']}:{p['debug_line']}: "
            f"{row['current_x_final_direction']} -> "
            f"{row['candidate_final_direction']} ({row['before_reason']} -> "
            f"{row['after_reason']})\n"
        )
    text += (
        '\nComplete rows and shared requests: comparison_rows.jsonl.gz. '
        'Exclusions: comparison.json.\n'
    )
    write_new(output / 'comparison.md', text)


class FrozenEngine:
    """Transport full historical results, audit requests, never call search()."""

    def __init__(self, case):
        from snake_bot.search_engine import SearchResult

        self.case = case
        rows = case['diagnostic'].get('deep_moves', [])
        first = rows[0] if rows else {}
        self.result = SearchResult(
            scores={
                r['direction']: r['search_score']
                for r in rows
                if r.get('search_score') is not None
            },
            completed_depth=first.get('search_depth', 0),
            nodes=first.get('search_nodes', 0),
            transposition_hits=first.get('search_transposition_hits', 0),
            elapsed_ms=first.get('search_elapsed_ms', 0),
        )
        self.request = None
        self.calls = 0

    def from_game_state(self, state, time_budget_ms=90, max_depth=6):
        from snake_bot.search_engine import IterativeSearchEngine

        expected, _ = state_for(self.case)
        _, compact = IterativeSearchEngine.from_game_state(state, time_budget_ms, max_depth)
        _, expected_compact = IterativeSearchEngine.from_game_state(
            expected, time_budget_ms, max_depth
        )
        if compact != expected_compact:
            raise ValueError('Recorded state mismatch for frozen search')

        def search(actual_compact, side, legal, target=None):
            request = normalized(
                {
                    'state': asdict(actual_compact),
                    'side': side,
                    'legal': list(legal),
                    'target': target,
                    'budget': time_budget_ms,
                    'max_depth': max_depth,
                }
            )
            self.verify_request(
                request, expected, expected_compact, side, legal, target, time_budget_ms
            )
            self.request = request
            self.calls += 1
            return copy.deepcopy(self.result)

        return SimpleNamespace(search=search), compact

    def verify_request(self, request, expected, expected_compact, side, legal, target, budget):
        """Require both replay consistency and the recorded historical input fields."""
        if self.request is not None and request != self.request:
            raise ValueError('Frozen request mismatch between replay versions')
        if not self.matches_historical_request(
            request, expected, expected_compact, side, legal, target, budget
        ):
            raise ValueError('Historical request mismatch: target/legal/budget/state')

    def matches_historical_request(
        self, request, expected, expected_compact, side, legal, target, budget
    ):
        historical_target = (self.case['diagnostic'].get('plan') or {}).get('food')
        actual = {
            'state': request['state'],
            'side': side,
            'legal': set(legal),
            'target': normalized(target),
            'budget': budget,
        }
        recorded = {
            'state': normalized(asdict(expected_compact)),
            'side': expected.side,
            'legal': set(self.case['diagnostic']['legal']),
            'target': historical_target,
            'budget': self.case['diagnostic'].get('search_time_budget_ms'),
        }
        return actual == recorded


def analyze_case(case):
    """Describe recorded score geometry without executing or changing strategy."""
    diagnostic = case['diagnostic']
    rows = diagnostic.get('deep_moves', [])
    depth = max((row.get('search_depth', 0) for row in rows), default=0)
    usable = usable_score_rows(rows)
    values = sorted((row['search_score'] for row in usable), reverse=True)
    all_values = recorded_scores(rows)
    clipped = [max(-200000, min(200000, value)) for value in values]
    eligible = score_depth_eligible(depth, values)
    differing = differing_eligible_scores(eligible, values)
    adjustments = legacy_adjustments(rows, depth)
    return {
        **raw_score_description(values),
        **recorded_adjustment_description(rows, usable, adjustments),
        **clipped_pair_description(values, clipped, eligible),
        'eligible_depth_and_two': eligible,
        'eligible_differing_raw': differing,
        'depth': depth,
        'terminal_class': terminal_class(all_values, diagnostic.get('remaining_moves'), depth),
        'extreme_sources_compatible': extreme_score_sources(all_values),
        'geometry': raw_score_geometry(values),
        'opposite_signs': opposite_score_signs(values),
        'inside_and_outside': inside_and_outside_scores(values),
        'all_clip_identical': differing and len(set(clipped)) == 1,
    }


def score_depth_eligible(depth, values):
    return depth >= 4 and len(values) >= 2


def differing_eligible_scores(eligible, values):
    return eligible and len(set(values)) >= 2


def usable_score_rows(rows):
    return [
        row for row in rows if row.get('search_score') is not None and not row.get('forced_loss')
    ]


def recorded_scores(rows):
    return [row['search_score'] for row in rows if row.get('search_score') is not None]


def raw_score_description(values):
    return {
        'raw_top1': values[0] if values else None,
        'raw_top2': second_raw_score(values),
        'raw_gap': raw_score_gap(values),
        'raw_spread': max(values) - min(values) if values else None,
    }


def second_raw_score(values):
    return values[1] if len(values) >= 2 else None


def raw_score_gap(values):
    return values[0] - values[1] if len(values) >= 2 else None


def extreme_score_sources(values):
    return sorted({extreme_score_source(value) for value in values if abs(value) >= 300000000})


def extreme_score_source(value):
    source = 'elimination_500m' if abs(value) >= 400000000 else 'final_300m'
    return source + ('_positive' if value > 0 else '_negative')


def raw_score_geometry(values):
    if not values:
        return 'no_raw'
    categories = (
        ('positive_above_200k', all(value > 200000 for value in values)),
        ('negative_below_minus200k', all(value < -200000 for value in values)),
        ('inside_200k', all(abs(value) <= 200000 for value in values)),
    )
    return first_score_category(categories)


def first_score_category(categories):
    return next((name for name, matches in categories if matches), 'inside_outside_or_opposite')


def opposite_score_signs(values):
    return bool(values and min(values) < 0 < max(values))


def inside_and_outside_scores(values):
    return any(abs(value) <= 200000 for value in values) and any(
        abs(value) > 200000 for value in values
    )


def recorded_adjustment_description(rows, usable, adjustments):
    return {
        'forced_loss_present': any(row.get('forced_loss') for row in rows),
        'SAFE2': safe_score_rows(usable) >= 2,
        'old_adjustments': adjustments,
        'actual_adjustments_match': all(
            row.get('search_adjustment', 0) == adjustments[row['direction']] for row in rows
        ),
    }


def safe_score_rows(rows):
    return sum(row.get('safety') == 'SAFE' and not row.get('forced_loss') for row in rows)


def clipped_pair_description(values, clipped, eligible):
    center = sum(clipped) // len(clipped) if clipped else 0
    uncapped = [(value - center) * 35 // 100 for value in clipped]
    capped = [max(-25000, min(25000, value)) for value in uncapped]
    return clipped_pair_metrics(values, clipped, uncapped, capped, eligible)


def clipped_pair_metrics(values, clipped, uncapped, capped, eligible):
    return {
        'clip_masked_pairs': masked_score_pairs(values, clipped, eligible),
        'final_cap_masked_pairs': masked_score_pairs(uncapped, capped, eligible),
        'old_cap_active': eligible and any(abs(value) > 25000 for value in uncapped),
        'old_floor_masked_pairs': masked_score_pairs(clipped, uncapped, eligible),
    }


def masked_score_pairs(before, after, eligible):
    if not eligible:
        return 0
    pairs = ((i, j) for i in range(len(before)) for j in range(i + 1, len(before)))
    return sum(score_pair_masked(before, after, i, j) for i, j in pairs)


def score_pair_masked(before, after, first, second):
    return before[first] != before[second] and after[first] == after[second]


def distribution(values):
    """Summarize recorded numeric values with fixed histogram bins and ordered percentiles."""
    values = sorted(values)
    if not values:
        return {
            'n': 0,
            'min': None,
            'median': None,
            'p90': None,
            'p95': None,
            'max': None,
            'histogram': {},
        }
    bins = Counter(distribution_bin(value) for value in values)
    return {
        'n': len(values),
        'min': values[0],
        'median': median(values),
        'p90': values[(len(values) - 1) * 90 // 100],
        'p95': values[(len(values) - 1) * 95 // 100],
        'max': values[-1],
        'histogram': dict(sorted(bins.items())),
    }


def distribution_bin(value):
    if value == 0:
        return '0'
    thresholds = (
        (100, '1..100'),
        (1000, '101..1000'),
        (10000, '1001..10000'),
        (100000, '10001..100000'),
        (1000000, '100001..1000000'),
    )
    return next((name for limit, name in thresholds if value <= limit), '>1000000')


def summarize(cases):
    """Aggregate historical diagnostics while retaining every original denominator."""
    summary = CorpusSummary()
    for case in cases:
        summary.add(case)
    return summary.result(), summary.analyzed


class CorpusSummary:
    """Accumulate corpus counts, gap distributions and representative examples."""

    def __init__(self):
        self.counts, self.by_class, self.games = Counter(), defaultdict(Counter), defaultdict(set)
        self.gaps, self.differing_gaps, self.spreads = (
            defaultdict(list),
            defaultdict(list),
            defaultdict(list),
        )
        self.examples = defaultdict(list)
        self.observed_terminal_categories = Counter()
        self.analyzed = []

    def add(self, case):
        """Accumulate one recorded decision using the original eligibility denominators."""
        analysis = analyze_case(case)
        self.analyzed.append((case, analysis))
        self.counts['decisions'] += 1
        self.counts['actual_adjustment_mismatch'] += not analysis['actual_adjustments_match']
        self.counts['forced_loss_all_decisions'] += analysis['forced_loss_present']
        self.observed_terminal_categories[analysis['terminal_class']] += 1
        for marker in analysis['extreme_sources_compatible']:
            self.observed_terminal_categories[marker] += 1
        if analysis['eligible_depth_and_two']:
            self.add_eligible(case, analysis)

    def add_eligible(self, case, analysis):
        category = analysis['terminal_class']
        for key in eligible_analysis_flags(analysis):
            self.counts[key] += 1
            self.by_class[category][key] += 1
            self.games[key].add(case['provenance']['game_id'])
        for key in ('clip_masked_pairs', 'final_cap_masked_pairs', 'old_floor_masked_pairs'):
            self.counts[key] += analysis[key]
            self.by_class[category][key] += analysis[key]
        for source in analysis['extreme_sources_compatible']:
            self.counts[source] += 1
        self.record_distributions(case, analysis)

    def record_distributions(self, case, analysis):
        category = analysis['terminal_class']
        self.gaps[category].append(analysis['raw_gap'])
        if analysis['eligible_differing_raw']:
            self.differing_gaps[category].append(analysis['raw_gap'])
        self.spreads[category].append(analysis['raw_spread'])
        if analysis['all_clip_identical']:
            self.record_all_clip(case, analysis)

    def record_all_clip(self, case, analysis):
        diagnostic = case['diagnostic']
        enemy = 'B' if diagnostic['side'] == 'A' else 'A'
        lead = diagnostic['scores'][diagnostic['side']] - diagnostic['scores'][enemy]
        key = all_clip_lead_key(lead)
        category = analysis['terminal_class']
        self.counts[key] += 1
        self.by_class[category][key] += 1
        self.games[category + '/all_clip_identical'].add(case['provenance']['game_id'])
        self.examples[category + '/' + analysis['geometry']].append(
            {
                'provenance': case['provenance'],
                'analysis': analysis,
                'historic_direction': diagnostic['direction'],
            }
        )

    def result(self):
        """Return the stable aggregate schema for counts, distributions and examples."""
        return {
            'counts': dict(self.counts),
            'by_terminal_class': {key: dict(values) for key, values in self.by_class.items()},
            'observed_terminal_categories_all_decisions': dict(self.observed_terminal_categories),
            'games_with_at_least_one': {key: len(values) for key, values in self.games.items()},
            'raw_top1_top2_gap': distribution_mapping(self.gaps),
            'raw_top1_top2_gap_differing_raw': distribution_mapping(self.differing_gaps),
            'raw_spread': distribution_mapping(self.spreads),
            'top_examples': top_corpus_examples(self.examples),
            'candidate_formula_status': 'not approved; no hypothetical candidate computed',
        }


def eligible_analysis_flags(analysis):
    flags = (
        'eligible_differing_raw',
        'all_clip_identical',
        'forced_loss_present',
        'SAFE2',
        'opposite_signs',
        'inside_and_outside',
        'old_cap_active',
    )
    return ['eligible_depth_and_two', analysis['geometry']] + [
        flag for flag in flags if analysis[flag]
    ]


def all_clip_lead_key(lead):
    if lead > 0:
        return 'allclip_ahead'
    return 'allclip_behind' if lead < 0 else 'allclip_tied'


def distribution_mapping(values):
    return {key: distribution(items) for key, items in values.items()}


def top_corpus_examples(examples):
    top = []
    for key, rows in sorted(examples.items()):
        top.extend(sorted(rows, key=lambda row: row['analysis']['raw_gap'], reverse=True)[:3])
    return top


def select_fixtures(analyzed, minimum=24):
    """Keep category coverage and historical special cases before filling by score gap."""
    selection = FixtureSelection()
    selection.cover_categories(analyzed)
    selection.fill_by_gap(analyzed, minimum)
    return selection.chosen


class FixtureSelection:
    """Deduplicate selected observations while preserving the original selection order."""

    def __init__(self):
        self.chosen, self.seen, self.categories = [], set(), set()

    def take(self, case, analysis):
        """Add a unique observed decision while preserving its selection order."""
        identity = (case['provenance']['filename'], case['provenance']['debug_line'])
        if identity not in self.seen:
            self.seen.add(identity)
            self.chosen.append(dict(case, analysis=analysis))

    def cover_categories(self, analyzed):
        """Select existing category representatives and required historical edge cases."""
        for case, analysis in analyzed:
            category = fixture_category(analysis)
            if special_historical_fixture(case) or (
                case['diagnostic'].get('deep_moves') and category not in self.categories
            ):
                self.categories.add(category)
                self.take(case, analysis)

    def fill_by_gap(self, analyzed, minimum):
        """Fill the remaining fixture quota with eligible large-gap observations."""
        for case, analysis in sorted(
            analyzed, key=lambda pair: pair[1]['raw_gap'] or 0, reverse=True
        ):
            if len(self.chosen) >= minimum:
                break
            if analysis['eligible_differing_raw']:
                self.take(case, analysis)


def fixture_category(analysis):
    return (
        analysis['terminal_class'],
        analysis['geometry'],
        analysis['old_cap_active'],
        analysis['forced_loss_present'],
        tuple(analysis['extreme_sources_compatible']),
        analysis['SAFE2'],
        analysis['opposite_signs'],
    )


def special_historical_fixture(case):
    provenance = case['provenance']
    return (
        provenance['game_id'].startswith('8d25279e')
        and provenance['debug_line'] == 498
        or provenance['game_id'].startswith('5580bbd6')
        and provenance['debug_line'] == 576
    )


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8') as stream:
        stream.write(
            value if isinstance(value, str) else json.dumps(value, indent=2, sort_keys=True) + '\n'
        )


def markdown(summary):
    lines = [
        '# Deep clipping phase 1: frozen historical corpus',
        '',
        ('No candidate normalization, production edits, timed searches or ' 'games.'),
        '',
        (
            'Denominators: depth>=4 and >=2 non-forced raw scores includes '
            'raw ties; differing raw is a separate subset.'
        ),
        (
            'Terminal-horizon: 0<=R<=completed depth. Extreme: abs(any '
            'raw)>=300M; compatible, not proof of terminal origin.'
        ),
        (
            'Source returns +/-500M plus score delta for elimination, +/-300M '
            'plus delta for final score; zero can mean a terminal draw.'
        ),
        '',
        '| Metric | Count |',
        '|---|---:|',
    ]
    lines.extend(f'| {k} | {v} |' for k, v in summary['counts'].items())
    lines += [
        '',
        '## By Terminal Class',
        '',
        '```json',
        json.dumps(summary['by_terminal_class'], indent=2),
        '```',
        '',
        '## Real Observed Terminal Categories',
        '',
        '```json',
        json.dumps(summary['observed_terminal_categories_all_decisions'], indent=2),
        '```',
        '',
        (
            'Absent categories are not fabricated. Marker families are '
            'magnitude-compatible, not traced provenance.'
        ),
        '',
        '## Top1/Top2 Raw Gaps',
        '',
        ('Quantiles use lower-index empirical order statistics; median is ' 'the usual midpoint.'),
        '',
        '```json',
        json.dumps(summary['raw_top1_top2_gap'], indent=2),
        '```',
        '',
        '## Games With At Least One',
        '',
        '```json',
        json.dumps(summary['games_with_at_least_one'], indent=2),
        '```',
        '',
        '## Examples',
        '',
    ]
    for row in summary['top_examples']:
        p, a = row['provenance'], row['analysis']
        lines.append(
            (
                f"- {p['filename']}:{p['debug_line']} {a['terminal_class']}, "
                f"raw {a['raw_top1']}/{a['raw_top2']}, gap {a['raw_gap']}, "
                f"historic move {row['historic_direction']}"
            )
        )
    return '\n'.join(lines) + '\n'


def main(argv=None):
    """Dispatch the requested historical evidence command after validating its required inputs."""
    parser, args = parse_replay_arguments(argv)
    standalone = {'compact': compact_command, 'replay': replay_command}
    handler = standalone.get(args.command)
    if handler is not None:
        return handler(args)
    return cohort_command(parser, args)


def cohort_command(parser, args):
    if args.report is None or args.frozen_manifest is None:
        parser.error('analyze requires --report and --frozen-manifest')
    cases = load_cohort(args.report, args.frozen_manifest, args.games)
    if args.command == 'compare':
        return compare_command(args, cases)
    frozen = json.loads(args.frozen_manifest.read_text())
    names = manifest_names(args.report)
    handlers = {
        'hypothesis': hypothesis_command,
        'details': details_command,
        'analyze': analyze_command,
    }
    return handlers[args.command](args, cases, frozen, names)


def parse_replay_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'command',
        choices=(
            'analyze',
            'compact',
            'replay',
            'hypothesis',
            'details',
            'compare',
        ),
    )
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
    return parser, args


def compact_command(args):
    fixture = json.loads(args.fixtures.read_text())
    histories = {
        c['provenance']['game_id'] + ':' + str(c['provenance']['debug_line']): c['history']
        for c in fixture['cases']
    }
    if any(isinstance(c['history'], list) for c in fixture['cases']):
        with gzip.open(args.output / 'fixture_histories.json.gz', 'xb') as stream:
            stream.write(json.dumps(histories, sort_keys=True).encode())
    fixture['cases'] = [compact_fixture_history(c) for c in fixture['cases']]
    temporary = args.fixtures.with_suffix('.compact.json')
    write_new(
        temporary,
        json.dumps(fixture, sort_keys=True, separators=(',', ':')) + '\n',
    )
    temporary.replace(args.fixtures)
    print(
        json.dumps(
            {
                'fixture_count': len(fixture['cases']),
                'bytes': args.fixtures.stat().st_size,
            }
        )
    )
    return


def replay_command(args):
    source = load_baseline(args.baseline_search)
    fixture = json.loads(args.fixtures.read_text())
    rows = [replay_current_reference(c, source) for c in fixture['cases']]
    counts = reference_replay_counts(rows)
    game_counts = reference_replay_games(rows)
    result = {
        'counts': dict(counts),
        'pool': 'selected real fixtures; not representative corpus sample',
        'games_with_at_least_one': game_counts,
        'historical_request_certified_count': 0,
        'results': rows,
    }
    write_new(args.output / 'reference_replay.json', result)
    write_new(
        args.output / 'reference_replay.md',
        '# Current X64 frozen-score reference replay\n\n```json\n'
        + json.dumps(dict(counts), indent=2)
        + (
            '\n```\n\nNo candidate brain executed. Changed-target cases '
            'excluded. Historic source/request not certified.\n'
        ),
    )
    print(json.dumps(dict(counts), indent=2))


def reference_replay_counts(rows):
    counts = Counter(row['status'] for row in rows)
    for flag in (
        'historic_vs_current_x_final_changed',
        'current_x_plan_changed',
        'historic_vs_current_x_adjustment_changed',
        'historic_vs_current_x_order_changed',
        'safe_repair',
        'raw_result_preserved',
        'shared_request_and_result_exact',
        'hunger_exact',
        'repeat_count_exact',
    ):
        counts[flag] = sum(bool(r.get(flag)) for r in rows)
    return counts


def reference_replay_games(rows):
    return {
        flag: len({r['provenance']['game_id'] for r in rows if r.get(flag)})
        for flag in (
            'historic_vs_current_x_final_changed',
            'historic_vs_current_x_order_changed',
        )
    }


def compare_command(args, cases):
    baseline = load_baseline(args.baseline_search)
    candidate = load_source(args.candidate_search, args.candidate_sha256)
    result = run_comparison(cases, baseline, candidate, args.output, args.limit)
    print(json.dumps(result['counts'], indent=2))
    return


def hypothesis_command(args, cases, frozen, names):
    source = load_baseline(args.baseline_search)
    result = measure_hypothesis(cases, source)
    write_new(args.output / 'hypothesis.json', result)
    write_new(
        args.output / 'hypothesis.md',
        '# Approved formula: hypothetical historical ranking impact\n\n'
        + result['scope']
        + '\n\n'
        + result['formula']
        + '\n\n```json\n'
        + json.dumps(
            {k: v for k, v in result.items() if k not in ('top_examples', 'cap_examples')},
            indent=2,
        )
        + (
            '\n```\n\nUnchanged rankings are not attributed to cap except '
            'for the explicit b control. Full example rows in '
            'hypothesis.json.\n'
        ),
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k not in ('top_examples', 'cap_examples')},
            indent=2,
        )
    )
    return


def details_command(args, cases, frozen, names):
    summary, _ = summarize(cases)
    write_new(args.output / 'corpus_details.json', summary)
    write_new(args.output / 'corpus_details.md', markdown(summary))
    print(
        json.dumps(
            {
                'counts': summary['counts'],
                'terminal_categories': summary['observed_terminal_categories_all_decisions'],
                'gaps_differing_raw': summary['raw_top1_top2_gap_differing_raw'],
            },
            indent=2,
        )
    )
    return


def analyze_command(args, cases, frozen, names):
    summary, analyzed = summarize(cases)
    summary['manifest'] = {
        'report_sha256': frozen['report_sha256'],
        'filenames': names,
        'log_hashes': frozen['log_hashes'],
        'historical_declared_version': '2.16',
        'historical_planner': 'pre-X fix; not cryptographically certified',
        'current_reference_x_sha256': REFERENCE_X,
        'move_search_baseline_sha256': REFERENCE_SEARCH,
    }
    fixtures = select_fixtures(analyzed)
    summary['fixture_count'] = len(fixtures)
    summary['fixture_categories'] = dict(
        Counter(
            f['analysis']['terminal_class'] + '/' + f['analysis']['geometry'] for f in fixtures
        )
    )
    histories = {
        c['provenance']['game_id'] + ':' + str(c['provenance']['debug_line']): c['history']
        for c in fixtures
    }
    with gzip.open(args.output / 'fixture_histories.json.gz', 'xb') as stream:
        stream.write(json.dumps(histories, sort_keys=True).encode())
    compacted = [compact_fixture_history(c) for c in fixtures]
    write_new(
        args.fixtures,
        json.dumps(
            {'manifest': summary['manifest'], 'cases': compacted},
            sort_keys=True,
            separators=(',', ':'),
        )
        + '\n',
    )
    write_new(args.output / 'corpus.json', summary)
    write_new(args.output / 'corpus.md', markdown(summary))
    print(
        json.dumps(
            {
                'counts': summary['counts'],
                'by_terminal_class': summary['by_terminal_class'],
                'raw_gaps': summary['raw_top1_top2_gap'],
                'fixtures': summary['fixture_categories'],
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
