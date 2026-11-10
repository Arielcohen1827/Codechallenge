"""Paired weight comparisons using the existing self-play simulator."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

from snake_bot import bot_weights
from tools.bot_lab import (
    MECHANICS_VALIDATION_STATUS,
    V7_METRICS,
    simulate_game,
)
from snake_bot.bot_version import BOT_VERSION
from snake_bot.game_rules import MAX_MOVES

SOURCE_FILES = (
    'tools/benchmark_bot.py',
    'tools/bot_lab.py',
    'snake_bot/bot_weights.py',
    'snake_bot/bot_version.py',
    'snake_bot/snake_brain.py',
    'snake_bot/food_planner.py',
    'snake_bot/move_search.py',
    'snake_bot/search_engine.py',
    'snake_bot/snake_state.py',
    'snake_bot/game_rules.py',
)
EVENTS = {
    'correct_foods': 'foods',
    'wrong_digits': 'wrong_digits',
    'pickups': 'pickups',
    'wall_hits': 'wall_hits',
    'control_moves': 'control_moves',
}
DIAGNOSTIC_EVENTS = tuple(key[:-2] for key in V7_METRICS if key.endswith('_A'))
ALL_EVENTS = (*EVENTS, *DIAGNOSTIC_EVENTS)


def canonical_hash(value):
    """Hash canonical JSON so equivalent effective configurations share an identity."""
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def load_configuration(path):
    """Validate a weight file and record both supplied and effective configurations."""
    path = Path(path).resolve()
    try:
        if path.suffix.lower() != '.json':
            raise ValueError()
        raw_bytes = path.read_bytes()
        raw = json.loads(raw_bytes)
        if not isinstance(raw, dict):
            raise ValueError()
        effective = bot_weights.load_weights(path)
        return {
            'source_path': str(path),
            'source_sha256': hashlib.sha256(raw_bytes).hexdigest(),
            'effective_weights': effective,
            'effective_sha256': canonical_hash(effective),
            'ignored_key_count': len(raw.keys() - bot_weights.DEFAULT_WEIGHTS.keys()),
        }
    except (OSError, ValueError, TypeError, OverflowError):
        raise ValueError(
            (
                'Cannot load weight configuration as a JSON object with '
                'integer-convertible known weights.'
            )
        ) from None


def git_snapshot():
    """Capture repository revision and dirty state when Git metadata is available."""
    try:
        root = Path(__file__).resolve().parent.parent
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        status = subprocess.run(
            ['git', 'status', '--porcelain'], cwd=root, check=True, capture_output=True, text=True
        ).stdout
        return {'commit': commit, 'dirty': bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'dirty': None}


def normalize_game(result, seed, repetition, candidate_side):
    """Orient a simulator result to the candidate while preserving raw side attribution."""
    if candidate_side not in ('A', 'B'):
        raise ValueError('Candidate side must be A or B.')
    reference_side = 'B' if candidate_side == 'A' else 'A'
    participants = {candidate_side: 'candidate', reference_side: 'reference'}
    scores = {'A': result.score_a, 'B': result.score_b}
    return {
        'seed': seed,
        'repetition': repetition,
        'candidate_side': candidate_side,
        'participants': participants,
        'winner': result.winner,
        'winner_participant': participants.get(result.winner),
        'scores': scores,
        'candidate_score': scores[candidate_side],
        'reference_score': scores[reference_side],
        'candidate_score_diff': scores[candidate_side] - scores[reference_side],
        'turns': result.turns,
        'events': events_by_participant(result, participants),
        'termination_reason': getattr(result, 'termination_reason', None),
        'move_budget': getattr(result, 'move_budget', None),
        'remaining_moves': getattr(result, 'remaining_moves', None),
        'game_wall_time_seconds': None,
        'global_diagnostics': {
            name: getattr(result, name, None) for name in ('edge_moves', 'center_score_total')
        },
        'no_legal_side': result.no_legal_side,
        'no_legal_participant': participants.get(result.no_legal_side),
    }


def events_by_participant(result, participants):
    """Map full-truth side events to candidate and reference participants."""
    diagnostics = getattr(result, 'diagnostics', {}) or {}
    return {
        participants[side]: participant_events(result, diagnostics, side) for side in ('A', 'B')
    }


def participant_events(result, diagnostics, side):
    """Collect one side's resource and rule diagnostics using the stable schema."""
    return {
        **{
            name: getattr(result, f'{attribute}_{side.lower()}', 0)
            for name, attribute in EVENTS.items()
        },
        **{name: diagnostics.get(f'{name}_{side}', 0) for name in DIAGNOSTIC_EVENTS},
    }


def aggregate(games):
    """Combine outcomes, scores, rule events and timing for a selected set of games."""
    return {
        **benchmark_outcomes(games),
        **benchmark_scores(games),
        **benchmark_events(games),
        **benchmark_timing(games),
        'termination_reasons': dict(
            Counter(game.get('termination_reason') or 'unknown' for game in games)
        ),
    }


def benchmark_outcomes(games):
    """Count candidate wins, losses and draws with both win-rate definitions."""
    total = len(games)
    participants = Counter(game['winner_participant'] for game in games)
    wins = participants['candidate']
    losses = participants['reference']
    return {
        'total': total,
        'wins': wins,
        'losses': losses,
        'draws': total - wins - losses,
        'win_rate': ratio_or_none(wins, total),
        'win_rate_without_draws': ratio_or_none(wins, wins + losses),
    }


def ratio_or_none(numerator, denominator):
    """Leave a rate undefined when its denominator contains no observations."""
    return numerator / denominator if denominator else None


def mean_or_none(values):
    """Return a mean only when measurements are available."""
    return statistics.mean(values) if values else None


def median_or_none(values):
    """Return a median only when measurements are available."""
    return statistics.median(values) if values else None


def game_values(games, field):
    """Collect available measurements without treating missing timing as zero."""
    return [game[field] for game in games]


def benchmark_scores(games):
    """Summarize candidate-relative score differences and both participant scores."""
    differences = game_values(games, 'candidate_score_diff')
    return {
        'mean_score_diff': mean_or_none(differences),
        'median_score_diff': median_or_none(differences),
        'mean_scores': {
            role: mean_or_none(game_values(games, f'{role}_score'))
            for role in ('candidate', 'reference')
        },
        'mean_turns': mean_or_none(game_values(games, 'turns')),
    }


def benchmark_events(games):
    """Aggregate every stable event for each participant."""
    return {
        'mean_events': benchmark_event_statistics(games, mean_or_none),
        'total_events': benchmark_event_statistics(games, sum),
    }


def benchmark_event_statistics(games, aggregate_event):
    """Report total and per-game rates for one event across participants."""
    return {
        role: participant_event_statistics(games, role, aggregate_event)
        for role in ('candidate', 'reference')
    }


def participant_event_statistics(games, role, aggregate_event):
    """Collect one participant's event from the normalized game schema."""
    return {
        event: aggregate_event([game['events'][role].get(event, 0) for game in games])
        for event in ALL_EVENTS
    }


def benchmark_timing(games):
    """Measure simulator wall time and throughput from recorded per-game durations."""
    elapsed = [game.get('game_wall_time_seconds') for game in games]
    if not elapsed or None in elapsed:
        return {
            'total_game_wall_time_seconds': None,
            'seconds_per_game': None,
            'games_per_second': None,
            'median_game_wall_time_seconds': None,
        }
    wall_time = sum(elapsed)
    return {
        'total_game_wall_time_seconds': wall_time,
        'seconds_per_game': wall_time / len(elapsed),
        'games_per_second': ratio_or_none(len(elapsed), wall_time),
        'median_game_wall_time_seconds': statistics.median(elapsed),
    }


def summarize(games):
    """Report overall, per-side, per-repetition and paired-seed benchmark evidence."""
    groups = paired_games(games)
    pairs = [summarize_pair(seed, rep, slots) for (seed, rep), slots in groups.items()]
    return {
        **aggregate(games),
        'by_candidate_side': summaries_by_side(games),
        'by_repetition': summaries_by_repetition(games),
        'pairs': summarize_pairs(pairs),
    }


def games_by(games, key):
    """Group game records by an explicit schema field while preserving their order."""
    groups = {}
    for game in games:
        groups.setdefault(game[key], []).append(game)
    return groups


def summaries_by_side(games):
    """Keep candidate-as-A and candidate-as-B evidence separate."""
    groups = games_by(games, 'candidate_side')
    return {side: aggregate(groups.get(side, [])) for side in ('A', 'B')}


def summaries_by_repetition(games):
    """Report repeated-seed execution outcomes independently."""
    groups = games_by(games, 'repetition')
    return {str(rep): aggregate(groups[rep]) for rep in sorted(groups)}


def paired_games(games):
    """Index mirrored games by seed, repetition and candidate assignment."""
    grouped = {}
    for game in games:
        grouped.setdefault((game['seed'], game['repetition']), []).append(game)
    return grouped


def summarize_pair(seed, repetition, slots):
    """Describe one seed's mirrored pair and whether both assignments completed."""
    if len(slots) != 2 or {game['candidate_side'] for game in slots} != {'A', 'B'}:
        raise ValueError('Each seed/repetition pair must contain exactly candidate sides A and B.')
    diff_sum = sum(game_values(slots, 'candidate_score_diff'))
    return {
        **aggregate(slots),
        'seed': seed,
        'repetition': repetition,
        'candidate_sides': ['A', 'B'],
        'score_diff_sum': diff_sum,
        'score_diff_mean': diff_sum / 2,
        'by_candidate_side': summaries_by_side(slots),
    }


def summarize_pairs(pairs):
    """Aggregate complete mirrored pairs without counting incomplete evidence."""
    sums = game_values(pairs, 'score_diff_sum')
    means = game_values(pairs, 'score_diff_mean')
    return {
        'total': len(pairs),
        'records': pairs,
        **pair_score_signs(sums),
        'mean_score_diff_sum': mean_or_none(sums),
        'mean_score_diff_mean': mean_or_none(means),
        'median_score_diff_sum': median_or_none(sums),
        'median_score_diff_mean': median_or_none(means),
    }


def pair_score_signs(sums):
    """Count positive, negative and tied mirrored score sums."""
    return {
        'positive_score_diff_pairs': sum(value > 0 for value in sums),
        'negative_score_diff_pairs': sum(value < 0 for value in sums),
        'zero_score_diff_pairs': sum(value == 0 for value in sums),
    }


def validate_benchmark_dimensions(seeds, repetitions, turns, rows, cols, food):
    """Validate positive sample counts and the supported move budget."""
    if seeds <= 0 or repetitions <= 0:
        raise ValueError('Seeds and repetitions must be positive.')
    if type(turns) is not int or not 1 <= turns <= MAX_MOVES:
        raise ValueError(f'Turns must be an integer from 1 to {MAX_MOVES}.')
    validate_benchmark_board(rows, cols, food)


def validate_benchmark_board(rows, cols, food):
    """Reject unsupported board sizes and food quantities before simulation."""
    if rows < 0 or (cols != 0 and cols < 6) or not 1 <= food <= 9:
        raise ValueError('Rows must be 0 or >=1; cols must be 0 or >=6; food must be 1..9.')


def aliases_protected_path(output, protected_paths):
    """Detect existing output aliases to protected inputs, including hard links."""
    return output.exists() and any(
        path.exists() and output.samefile(path) for path in protected_paths
    )


def is_existing_game_log(output, output_parts):
    """Protect existing match logs while allowing the designated benchmark report directory."""
    if not output.exists():
        return False
    benchmark_directory = any(
        output_parts[index:index + 2] == ('games', 'benchmarks')
        for index in range(len(output_parts) - 1)
    )
    return 'logs' in output_parts or ('games' in output_parts and not benchmark_directory)


def has_reserved_report_location(output, output_parts):
    """Reject report names or directories reserved for live configurations."""
    return (
        output.suffix.lower() != '.json'
        or output.name.lower() in {'active_weights.json', 'best_weights.json'}
        or bool(
            set(output_parts) & {'weights', 'config', 'configs', 'configuration', 'configurations'}
        )
    )


def benchmark_report_path(output, reference, candidate, root):
    """Validate a report destination before any input or log can be overwritten."""
    output = Path(output).resolve()
    inputs = {Path(reference).resolve(), Path(candidate).resolve()}
    weights_directory = root / 'weights'
    protected_paths = inputs | {
        weights_directory / 'active_weights.json',
        weights_directory / 'best_weights.json',
    }
    output_parts = normalized_path_parts(output)
    if (
        has_reserved_report_location(output, output_parts)
        or output in protected_paths
        or aliases_protected_path(output, protected_paths)
        or is_existing_game_log(output, output_parts)
    ):
        raise ValueError(
            'Output must be a report JSON outside weight/configuration and game-log paths.'
        )
    return output


def normalized_path_parts(path):
    """Normalize report path components for platform-independent reserved names."""
    return tuple(part.lower() for part in path.parts)


def play_benchmark_game(game_seed, repetition, candidate_side, configurations, options):
    """Run and time one explicit candidate assignment without persisting weights."""
    reference_side = 'B' if candidate_side == 'A' else 'A'
    weights_by_side = {
        candidate_side: configurations['candidate']['effective_weights'],
        reference_side: configurations['reference']['effective_weights'],
    }
    started = perf_counter()
    result = simulate_game(
        seed=game_seed,
        weights_by_side=weights_by_side,
        candidate_side=candidate_side,
        **options,
    )
    elapsed = perf_counter() - started
    game = normalize_game(result, game_seed, repetition, candidate_side)
    game['game_wall_time_seconds'] = elapsed
    return game


def play_benchmark_games(sequence, repetitions, configurations, options):
    """Execute mirrored seeds in stable repetition, seed and side order."""
    return [
        play_benchmark_game(game_seed, repetition, side, configurations, options)
        for repetition in range(1, repetitions + 1)
        for game_seed in sequence
        for side in ('A', 'B')
    ]


def benchmark_options(turns, rows, cols, food):
    """Translate optional historical dimensions into simulator arguments."""
    return {'max_turns': turns, 'rows': rows or None, 'cols': cols or None, 'food_count': food}


def source_hashes(root):
    """Capture source digests and explicitly record unavailable files."""
    return {
        name: (
            hashlib.sha256((root / name).read_bytes()).hexdigest()
            if (root / name).is_file()
            else None
        )
        for name in SOURCE_FILES
    }


def write_benchmark_report(output, report):
    """Write finite JSON and surface filesystem failures as a report error."""
    serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
    try:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(serialized, encoding='utf-8')
    except OSError:
        raise ValueError('Cannot write benchmark JSON report.') from None


def run_benchmark(
    reference,
    candidate,
    seeds,
    seed,
    repetitions,
    output,
    turns=MAX_MOVES,
    rows=0,
    cols=0,
    food=5,
):
    """Compare two explicit configurations and restore active in-memory weights on every exit."""
    validate_benchmark_dimensions(seeds, repetitions, turns, rows, cols, food)
    root = Path(__file__).resolve().parent.parent
    output = benchmark_report_path(output, reference, candidate, root)
    configurations = {
        role: load_configuration(path)
        for role, path in [('reference', reference), ('candidate', candidate)]
    }
    sequence = list(range(seed, seed + seeds))
    manifest = {
        'schema_version': 2,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'bot_version': BOT_VERSION,
        'git': git_snapshot(),
        'start_seed': seed,
        'seed_count': seeds,
        'repetitions': repetitions,
        'seeds': sequence,
        'simulator_parameters': {
            'rows': rows,
            'cols': cols,
            'food_count': food,
            'max_turns': turns,
            'starting_side': 'A',
        },
        'official_max_moves': MAX_MOVES,
        'mechanics_validation_status': dict(MECHANICS_VALIDATION_STATUS),
        'comparison_priority': ['wins_losses_draws', 'score_difference'],
        'configurations': configurations,
        'effective_hashes_equal': configurations['reference']['effective_sha256']
        == configurations['candidate']['effective_sha256'],
        'source_sha256': source_hashes(root),
        'limitations': [
            (
                'Seeded initial states; wall-clock search budgets do not '
                'guarantee deterministic outcomes.'
            ),
            (
                'Self-play compares weights using the same current bot code '
                'for both participants.'
            ),
            (
                'Valid direction proposals are applied by bot_lab without '
                'move repairs; crashes are not automatic deaths.'
            ),
            (
                'Score elimination and movement limit use core v7 rules; a '
                'shorter max_turns is a simulation budget.'
            ),
            (
                'Crash and tail mechanics are implemented according to the '
                'official contract; pending live validation.'
            ),
            'Turns count global moves, not elapsed seconds.',
            (
                'Game wall time measures each simulate_game call with '
                'perf_counter, including both bots and simulation; it is not '
                'search time.'
            ),
            (
                'Throughput excludes configuration, report generation and '
                'file writing; per-search timing is unavailable.'
            ),
            (
                'Tail generated is attributed to its beneficiary; tail '
                'collection, cleaning and points are separate from numbered '
                'food.'
            ),
            'Rows/cols 0 select random dimensions from 20 through 30.',
        ],
    }
    options = benchmark_options(turns, rows, cols, food)
    previous = dict(bot_weights.ACTIVE_WEIGHTS)
    try:
        games = play_benchmark_games(sequence, repetitions, configurations, options)
        report = {
            'manifest': manifest,
            'summary': summarize(games),
            'games': games,
        }
        write_benchmark_report(output, report)
        return report
    finally:
        bot_weights.ACTIVE_WEIGHTS.clear()
        bot_weights.ACTIVE_WEIGHTS.update(previous)


def main(argv=None):
    """Parse benchmark inputs and write the requested comparison report."""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'candidate', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    for name in ('seeds', 'seed', 'repetitions'):
        parser.add_argument(f'--{name}', required=True, type=int)
    for name, default in [('turns', MAX_MOVES), ('rows', 0), ('cols', 0), ('food', 5)]:
        parser.add_argument(f'--{name}', default=default, type=int)
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(**vars(args))
    except ValueError as error:
        parser.error(str(error))
    stats = report['summary']
    print(
        f"games={stats['total']} wins={stats['wins']} losses={stats['losses']} "
        f"draws={stats['draws']} win_rate={stats['win_rate']:.4f} "
        f"mean_score_diff={stats['mean_score_diff']:.2f}"
    )
    print(f'report: {args.output}')


if __name__ == '__main__':
    main()
