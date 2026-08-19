import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from food_planner import choose_food_plan
from snake_brain import final_safe_direction
from snake_state import apply_move, in_bounds, legal_moves, parse_state, step


def parse_log(path):
    turns = []
    actions = []
    final = None
    metadata = {}
    decision_debug = []
    for line in path.read_text(errors='replace').splitlines():
        if line.startswith('= '):
            payload = json.loads(line[2:])
            if payload.get('event') == 'bot_version':
                metadata = payload
        elif line.startswith('< '):
            payload = json.loads(line[2:])
            if payload.get('event') == 'your_turn':
                turns.append(payload['data'])
            elif payload.get('event') == 'game_over':
                final = payload.get('data', {})
        elif line.startswith('> '):
            payload = json.loads(line[2:])
            if payload.get('action') == 'move':
                actions.append(payload.get('data', {}))
        elif line.startswith('? '):
            payload = json.loads(line[2:])
            if payload.get('event') == 'decision_debug':
                decision_debug.append(payload)
    return turns, actions, final, metadata, decision_debug


def score_food_delta(before, after, side):
    return after.scores.get(side, 0) - before.scores.get(side, 0) >= 100


def state_signature(state):
    return (
        tuple(state.snakes.get(state.side, ())),
        tuple(state.snakes.get(state.enemy, ())),
        tuple(sorted(state.food)),
    )


def analyze(path):
    turns, actions, final, metadata, decision_debug = parse_log(path)
    metrics = Counter()
    intervals = []
    last_food_turn = None
    first_food_move = None
    previous_scores = None
    repeated_seen = defaultdict(int)
    longest_cycle = 0
    previous_target = None
    target_abandoned = 0
    target_changes = 0
    debug_by_token = {
        debug.get('turn_token'): debug
        for debug in decision_debug
        if debug.get('turn_token') is not None
    }

    for index, data in enumerate(turns):
        state = parse_state(data)
        side = state.side
        action = actions[index] if index < len(actions) else {}
        direction = action.get('direction')
        if direction not in {'up', 'down', 'left', 'right'}:
            continue
        debug = debug_by_token.get(data.get('turn_token'))

        if previous_scores is not None:
            if state.scores.get('A', 0) - previous_scores.get('A', 0) >= 100:
                metrics['foods_A'] += 1
            if state.scores.get('B', 0) - previous_scores.get('B', 0) >= 100:
                metrics['foods_B'] += 1
        previous_scores = dict(state.scores)

        plan = choose_food_plan(state, side)
        target = plan.food if plan else None
        if previous_target is not None and target != previous_target and previous_target in state.food:
            target_changes += 1
            if plan and plan.safety in ('SAFE', 'ACCEPTABLE_RISK'):
                target_abandoned += 1
        previous_target = target

        sig = state_signature(state)
        repeated_seen[sig] += 1
        if repeated_seen[sig] > 1:
            metrics['repeated_states'] += 1
            longest_cycle = max(longest_cycle, repeated_seen[sig])

        legal = legal_moves(state, side)
        if direction not in legal:
            safe = final_safe_direction(data, direction)
            if safe != direction:
                metrics['illegal_moves_repaired'] += 1
            head = state.head(side)
            target = step(head, direction) if head is not None else None
            if not legal:
                metrics['no_legal_moves'] += 1
            if target is None or not in_bounds(state, target):
                metrics['out_of_bounds'] += 1
            else:
                target_char = state.board[target[0]][target[1]]
                if target_char == side.lower():
                    metrics['own_body_collision'] += 1
                elif target_char in (state.enemy, state.enemy.lower()):
                    metrics['rival_collision'] += 1

        head = state.head(side)
        adjacent_food_dirs = []
        if head is not None:
            for candidate in legal:
                if step(head, candidate) in state.food:
                    adjacent_food_dirs.append(candidate)
        if adjacent_food_dirs:
            metrics['adjacent_food_seen'] += 1
            if direction in adjacent_food_dirs:
                metrics['adjacent_food_taken'] += 1
            else:
                metrics['adjacent_food_rejected'] += 1
                for candidate in adjacent_food_dirs:
                    after_candidate = apply_move(state, candidate, side)
                    candidate_plan = choose_food_plan(state, side, step(head, candidate))
                    if candidate_plan is not None and candidate_plan.safety in ('SAFE', 'ACCEPTABLE_RISK'):
                        metrics['safe_adjacent_food_rejected'] += 1
                        break
                if debug:
                    plan = debug.get('plan') or {}
                    stats = debug.get('plan_move_stats') or {}
                    if plan.get('our_distance') == 1 and direction != plan.get('first_move'):
                        hard_pressure = (
                            stats.get('forced_zero')
                            or stats.get('forced_delayed_zero')
                            or stats.get('zero_escape_replies', 0) > 0
                        )
                        soft_pressure = (
                            stats.get('one_escape_replies', 0) >= 2
                            or stats.get('delayed_forced_zero_replies', 0) > 0
                        )
                        if hard_pressure:
                            metrics['hard_adjacent_food_rejected'] += 1
                        elif soft_pressure:
                            metrics['soft_adjacent_food_rejected'] += 1

        after = apply_move(state, direction, side)
        if score_food_delta(state, after, side):
            metrics['foods_us'] += 1
            if first_food_move is None:
                first_food_move = index
            if last_food_turn is not None:
                intervals.append(index - last_food_turn)
            last_food_turn = index
        elif last_food_turn is None:
            metrics['max_turns_without_food'] = max(metrics['max_turns_without_food'], index + 1)
        else:
            metrics['max_turns_without_food'] = max(metrics['max_turns_without_food'], index - last_food_turn)

        metrics['self_collisions'] = metrics['own_body_collision']

    metrics['total_turns'] = len(turns)
    if final:
        final_a = final.get('score_1')
        final_b = final.get('score_2')
        if final_a is not None:
            metrics['final_score_A'] = int(final_a)
        if final_b is not None:
            metrics['final_score_B'] = int(final_b)
    metrics['first_food_move'] = -1 if first_food_move is None else first_food_move
    metrics['average_food_interval'] = 0 if not intervals else round(sum(intervals) / len(intervals), 2)
    metrics['food_difference'] = metrics['foods_A'] - metrics['foods_B']
    metrics['food_per_100_moves'] = 0 if not turns else round(metrics['foods_us'] * 100 / len(turns), 2)
    metrics['target_changes'] = target_changes
    metrics['target_abandoned'] = target_abandoned
    metrics['longest_cycle'] = longest_cycle
    metrics['self_traps'] = metrics['safe_adjacent_food_rejected']
    metrics['final'] = final or {}
    metrics['bot_version'] = metadata.get('version', 'unknown')
    metrics['bot_notes'] = metadata.get('notes', '')
    metrics['decision_debug_lines'] = len(decision_debug)
    return metrics


def print_report(path, metrics):
    print(path)
    for key in [
        'total_turns',
        'bot_version',
        'decision_debug_lines',
        'foods_us',
        'foods_A',
        'foods_B',
        'food_difference',
        'final_score_A',
        'final_score_B',
        'food_per_100_moves',
        'first_food_move',
        'average_food_interval',
        'max_turns_without_food',
        'adjacent_food_seen',
        'adjacent_food_taken',
        'adjacent_food_rejected',
        'safe_adjacent_food_rejected',
        'soft_adjacent_food_rejected',
        'hard_adjacent_food_rejected',
        'target_changes',
        'target_abandoned',
        'repeated_states',
        'longest_cycle',
        'self_traps',
        'self_collisions',
        'own_body_collision',
        'out_of_bounds',
        'no_legal_moves',
        'rival_collision',
        'illegal_moves_repaired',
    ]:
        print(f"  {key}: {metrics[key]}")
    if metrics['bot_notes']:
        print(f"  bot_notes: {metrics['bot_notes']}")
    if metrics['final']:
        print(f"  final: {metrics['final']}")


def expand_log_paths(raw_paths):
    paths = []
    for raw_path in raw_paths:
        path = Path(raw_path)
        matches = sorted(Path('.').glob(raw_path)) if any(ch in raw_path for ch in '*?[') else []
        if not matches and any(ch in raw_path for ch in '*?[') and path.parent == Path('.'):
            matches = sorted(Path('games').glob(raw_path))
        if matches:
            paths.extend(matches)
        elif path.exists():
            paths.append(path)
        elif Path('games', raw_path).exists():
            paths.append(Path('games', raw_path))
        else:
            paths.append(path)
    return paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='+')
    args = parser.parse_args()
    for path in expand_log_paths(args.logs):
        print_report(path, analyze(path))


if __name__ == '__main__':
    main()
