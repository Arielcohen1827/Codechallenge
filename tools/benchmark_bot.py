"""Paired weight comparisons using the existing self-play simulator."""

import argparse
import hashlib
import json
import platform
import statistics
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from snake_bot import bot_weights
from tools.bot_lab import simulate_game
from snake_bot.bot_version import BOT_VERSION


SOURCE_FILES = (
    'tools/benchmark_bot.py', 'tools/bot_lab.py',
    'snake_bot/bot_weights.py', 'snake_bot/bot_version.py',
    'snake_bot/snake_brain.py', 'snake_bot/food_planner.py',
    'snake_bot/move_search.py', 'snake_bot/search_engine.py',
    'snake_bot/snake_state.py', 'snake_bot/game_rules.py',
)
EVENTS = {'correct_foods': 'foods', 'wrong_digits': 'wrong_digits',
          'pickups': 'pickups', 'wall_hits': 'wall_hits'}


def canonical_hash(value):
    payload = json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def load_configuration(path):
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
        raise ValueError('Cannot load weight configuration as a JSON object with integer-convertible known weights.') from None


def git_snapshot():
    try:
        root = Path(__file__).resolve().parent.parent
        commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, check=True,
                                capture_output=True, text=True).stdout.strip()
        status = subprocess.run(['git', 'status', '--porcelain'], cwd=root, check=True,
                                capture_output=True, text=True).stdout
        return {'commit': commit, 'dirty': bool(status.strip())}
    except (OSError, subprocess.SubprocessError):
        return {'commit': None, 'dirty': None}


def normalize_game(result, seed, repetition, candidate_side):
    reference_side = 'B' if candidate_side == 'A' else 'A'
    participants = {candidate_side: 'candidate', reference_side: 'reference'}
    scores = {'A': result.score_a, 'B': result.score_b}
    events = {
        participants[side]: {name: getattr(result, f'{attribute}_{side.lower()}')
                             for name, attribute in EVENTS.items()}
        for side in ('A', 'B')
    }
    return {
        'seed': seed, 'repetition': repetition, 'candidate_side': candidate_side,
        'participants': participants, 'winner': result.winner,
        'winner_participant': participants.get(result.winner), 'scores': scores,
        'candidate_score': scores[candidate_side], 'reference_score': scores[reference_side],
        'candidate_score_diff': scores[candidate_side] - scores[reference_side],
        'turns': result.turns, 'events': events,
        'no_legal_side': result.no_legal_side,
        'no_legal_participant': participants.get(result.no_legal_side),
    }


def aggregate(games):
    total = len(games)
    wins = sum(game['winner_participant'] == 'candidate' for game in games)
    losses = sum(game['winner_participant'] == 'reference' for game in games)
    diffs = [game['candidate_score_diff'] for game in games]
    def mean(values):
        return statistics.mean(values) if total else None
    return {
        'total': total, 'wins': wins, 'losses': losses, 'draws': total - wins - losses,
        'win_rate': wins / total if total else None,
        'win_rate_without_draws': wins / (wins + losses) if wins + losses else None,
        'mean_score_diff': mean(diffs),
        'median_score_diff': statistics.median(diffs) if total else None,
        'mean_scores': {role: mean([game[f'{role}_score'] for game in games])
                        for role in ('candidate', 'reference')},
        'mean_turns': mean([game['turns'] for game in games]),
        'mean_events': {role: {event: mean([game['events'][role][event] for game in games])
                               for event in EVENTS} for role in ('candidate', 'reference')},
    }


def summarize(games):
    summary = aggregate(games)
    summary['by_candidate_side'] = {
        side: aggregate([game for game in games if game['candidate_side'] == side])
        for side in ('A', 'B')
    }
    repetitions = sorted({game['repetition'] for game in games})
    summary['by_repetition'] = {
        str(rep): aggregate([game for game in games if game['repetition'] == rep])
        for rep in repetitions
    }
    grouped = {}
    for game in games:
        grouped.setdefault((game['seed'], game['repetition']), []).append(game)
    pairs = []
    for (seed, repetition), slots in grouped.items():
        if len(slots) != 2 or {game['candidate_side'] for game in slots} != {'A', 'B'}:
            raise ValueError('Each seed/repetition pair must contain exactly candidate sides A and B.')
        stats = aggregate(slots)
        diff_sum = sum(game['candidate_score_diff'] for game in slots)
        pairs.append({'seed': seed, 'repetition': repetition, 'candidate_sides': ['A', 'B'],
                      'score_diff_sum': diff_sum, 'score_diff_mean': diff_sum / 2,
                      'wins': stats['wins'], 'losses': stats['losses'], 'draws': stats['draws']})
    summary['pairs'] = {
        'total': len(pairs), 'records': pairs,
        'positive_score_diff_pairs': sum(pair['score_diff_sum'] > 0 for pair in pairs),
        'negative_score_diff_pairs': sum(pair['score_diff_sum'] < 0 for pair in pairs),
        'zero_score_diff_pairs': sum(pair['score_diff_sum'] == 0 for pair in pairs),
        'mean_score_diff_sum': statistics.mean([p['score_diff_sum'] for p in pairs]) if pairs else None,
        'mean_score_diff_mean': statistics.mean([p['score_diff_mean'] for p in pairs]) if pairs else None,
    }
    return summary


def run_benchmark(reference, candidate, seeds, seed, repetitions, output,
                  turns=300, rows=0, cols=0, food=5):
    if seeds <= 0 or repetitions <= 0 or turns <= 0:
        raise ValueError('Seeds, repetitions and turns must be positive.')
    if rows < 0 or (cols != 0 and cols < 6) or not 1 <= food <= 9:
        raise ValueError('Rows must be 0 or >=1; cols must be 0 or >=6; food must be 1..9.')
    output = Path(output).resolve()
    inputs = {Path(reference).resolve(), Path(candidate).resolve()}
    protected_names = {'active_weights.json', 'best_weights.json'}
    output_parts = tuple(part.lower() for part in output.parts)
    benchmark_directory = any(
        output_parts[index:index + 2] == ('games', 'benchmarks')
        for index in range(len(output_parts) - 1)
    )
    if (output in inputs or output.suffix.lower() != '.json'
            or output.name.lower() in protected_names
            or 'weights' in output_parts
            or (output.exists() and ('logs' in output_parts
                                    or ('games' in output_parts and not benchmark_directory)))):
        raise ValueError('Output must be a report JSON outside weight/configuration and game-log paths.')
    configurations = {role: load_configuration(path)
                      for role, path in [('reference', reference), ('candidate', candidate)]}
    sequence = list(range(seed, seed + seeds))
    root = Path(__file__).resolve().parent.parent
    manifest = {
        'created_at': datetime.now(timezone.utc).isoformat(),
        'python_version': platform.python_version(), 'platform': platform.platform(),
        'bot_version': BOT_VERSION, 'git': git_snapshot(),
        'start_seed': seed, 'seed_count': seeds, 'repetitions': repetitions, 'seeds': sequence,
        'simulator_parameters': {'rows': rows, 'cols': cols, 'food_count': food,
                                 'max_turns': turns, 'starting_side': 'A'},
        'configurations': configurations,
        'effective_hashes_equal': configurations['reference']['effective_sha256'] == configurations['candidate']['effective_sha256'],
        'source_sha256': {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                          if (root / name).is_file() else None for name in SOURCE_FILES},
        'limitations': [
            'Seeded initial states; wall-clock search budgets do not guarantee deterministic outcomes.',
            'Self-play compares weights using the same current bot code for both participants.',
            'The simulator repairs illegal proposed moves; no_legal_side means no legal move was available.',
            'Turns count global moves, not elapsed seconds.',
            'Search diagnostics and elapsed total time are not exported by SimResult.',
            'Rows/cols 0 select random dimensions from 12 through 20.',
        ],
    }
    games = []
    previous = dict(bot_weights.ACTIVE_WEIGHTS)
    try:
        for repetition in range(1, repetitions + 1):
            for game_seed in sequence:
                for candidate_side in ('A', 'B'):
                    reference_side = 'B' if candidate_side == 'A' else 'A'
                    weights_by_side = {
                        candidate_side: configurations['candidate']['effective_weights'],
                        reference_side: configurations['reference']['effective_weights'],
                    }
                    result = simulate_game(seed=game_seed, max_turns=turns, rows=rows or None,
                                           cols=cols or None, food_count=food,
                                           weights_by_side=weights_by_side, candidate_side=candidate_side)
                    games.append(normalize_game(result, game_seed, repetition, candidate_side))
        report = {'manifest': manifest, 'summary': summarize(games), 'games': games}
        serialized = json.dumps(report, indent=2, allow_nan=False) + '\n'
        try:
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(serialized, encoding='utf-8')
        except OSError:
            raise ValueError('Cannot write benchmark JSON report.') from None
        return report
    finally:
        bot_weights.ACTIVE_WEIGHTS.clear()
        bot_weights.ACTIVE_WEIGHTS.update(previous)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('reference', 'candidate', 'output'):
        parser.add_argument(f'--{name}', required=True, type=Path)
    for name in ('seeds', 'seed', 'repetitions'):
        parser.add_argument(f'--{name}', required=True, type=int)
    for name, default in [('turns', 300), ('rows', 0), ('cols', 0), ('food', 5)]:
        parser.add_argument(f'--{name}', default=default, type=int)
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(**vars(args))
    except ValueError as error:
        parser.error(str(error))
    stats = report['summary']
    print(f"games={stats['total']} wins={stats['wins']} losses={stats['losses']} "
          f"draws={stats['draws']} win_rate={stats['win_rate']:.4f} "
          f"mean_score_diff={stats['mean_score_diff']:.2f}")
    print(f'report: {args.output}')


if __name__ == '__main__':
    main()
