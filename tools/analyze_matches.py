"""Chronological diagnostics for saved Snake protocol observations."""

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from snake_bot.food_planner import choose_food_plan
from snake_bot.geometry import directions_for
from snake_bot.snake_brain import final_safe_direction
from snake_bot.snake_state import (
    apply_move,
    in_bounds,
    legal_moves,
    parse_state,
    step,
)


def parse_log(path):
    """Read received snapshots, sent actions and local diagnostics in their original order."""
    turns, actions, debug = [], [], []
    singleton = {'bot_version': {}, 'game_over': None}
    handlers = {
        ('= ', 'bot_version'): lambda payload: singleton.update(bot_version=payload),
        ('< ', 'your_turn'): lambda payload: turns.append(payload['data']),
        ('< ', 'game_over'): lambda payload: singleton.update(game_over=payload.get('data', {})),
        ('? ', 'decision_debug'): debug.append,
        ('> ', 'move'): lambda payload: actions.append(dict(payload.get('data', {}))),
        ('> ', 'shoot'): lambda payload: actions.append(
            dict(payload.get('data', {}), action_type='shoot')
        ),
    }
    for line in path.read_text(encoding='utf-8', errors='replace').splitlines():
        read_log_record(line, handlers)
    return turns, actions, singleton['game_over'], singleton['bot_version'], debug


def read_log_record(line, handlers):
    """Dispatch one recognized log record while ignoring unrelated lines."""
    prefix = line[:2]
    if prefix not in {'= ', '< ', '> ', '? '}:
        return
    payload = json.loads(line[2:])
    key = 'action' if prefix == '> ' else 'event'
    handler = handlers.get((prefix, payload.get(key)))
    if handler:
        handler(payload)


def score_food_delta(before, after, side):
    """Recognize a food-sized positive score change in a simulated visible move."""
    return after.scores.get(side, 0) - before.scores.get(side, 0) >= 100


def state_signature(state):
    """Build a hashable observation signature retaining every coordinate axis and rule timer."""
    return (
        tuple(state.snakes.get(state.side, ())),
        tuple(state.snakes.get(state.enemy, ())),
        tuple(sorted(state.food)),
        tuple(sorted(state.food_values.items())),
        tuple(sorted(state.pickups)),
        tuple(sorted(state.multipliers.items())),
        tuple(sorted(state.walls)),
        tuple(sorted(state.shot_food)),
        tuple(sorted(state.tail_food_a)),
        tuple(sorted(state.tail_food_b)),
        tuple(sorted(state.bombs_large)),
        tuple(sorted(state.bombs_small)),
        tuple(sorted(state.stars)),
        state.invincible_a,
        state.invincible_b,
        state.star_frenzy,
        state.star_spawn_due_round,
        state.pending_star_spawns,
    )


def analyze(path):
    """Summarize action quality from chronological player observations."""
    turns, actions, final, metadata, decision_debug = parse_log(path)
    analysis = MatchAnalysis()
    debug_by_token = {
        d.get('turn_token'): d for d in decision_debug if d.get('turn_token') is not None
    }
    for index, data in enumerate(turns):
        action = actions[index] if index < len(actions) else {}
        analysis.observe(index, data, action, debug_by_token.get(data.get('turn_token')))
    return analysis.finish(turns, final, metadata, decision_debug)


class MatchAnalysis:
    """Accumulate match metrics without sharing temporary state between reports."""

    def __init__(self):
        """Create accumulators for one chronological match report."""
        self.metrics = Counter()
        self.intervals = []
        self.last_food_turn = None
        self.first_food_move = None
        self.previous_scores = None
        self.repeated_seen = defaultdict(int)
        self.longest_cycle = 0
        self.previous_target = None
        self.target_abandoned = 0
        self.target_changes = 0

    def observe(self, index, data, action, debug):
        """Consume one aligned observed turn and proposed action."""
        self.index, self.debug = (index, debug)
        self.state = parse_state(data)
        self.side = self.state.side
        self.direction = action.get('direction')
        if self.direction not in directions_for(self.state):
            return
        self.observe_scores()
        if action.get('action_type') == 'shoot':
            self.metrics['shoots_us'] += 1
            return
        self.observe_plan()
        self.observe_cycle()
        self.legal = legal_moves(self.state, self.side)
        self.observe_illegal(data)
        self.prepare_adjacent()
        self.observe_kinds()
        self.observe_adjacent()
        self.observe_special()
        self.observe_food()
        self.metrics['self_collisions'] = self.metrics['own_body_collision']

    def observe_scores(self):
        """Compare consecutive public score counters."""
        if self.previous_scores is not None:
            if self.state.scores.get('A', 0) - self.previous_scores.get('A', 0) >= 100:
                self.metrics['foods_A'] += 1
            if self.state.scores.get('B', 0) - self.previous_scores.get('B', 0) >= 100:
                self.metrics['foods_B'] += 1
        self.previous_scores = dict(self.state.scores)

    def observe_plan(self):
        """Track changes to currently available objectives."""
        plan = choose_food_plan(self.state, self.side)
        target = plan.food if plan else None
        if target_was_abandoned(self.previous_target, target, self.state):
            self.target_changes += 1
            if acceptable_plan(plan):
                self.target_abandoned += 1
        self.previous_target = target

    def observe_cycle(self):
        """Record repeated observed states."""
        sig = state_signature(self.state)
        self.repeated_seen[sig] += 1
        if self.repeated_seen[sig] > 1:
            self.metrics['repeated_states'] += 1
            self.longest_cycle = max(self.longest_cycle, self.repeated_seen[sig])

    def observe_illegal(self, data):
        """Classify proposals outside known legal moves."""
        if self.direction not in self.legal:
            safe = final_safe_direction(data, self.direction)
            if safe != self.direction:
                self.metrics['illegal_moves_repaired'] += 1
            self.head = self.state.head(self.side)
            target = step(self.head, self.direction) if self.head is not None else None
            if not self.legal:
                self.metrics['no_legal_moves'] += 1
            self.observe_collision(target)

    def observe_collision(self, target):
        """Name the known collision surface for one destination."""
        if target is None or not in_bounds(self.state, target):
            self.metrics['out_of_bounds'] += 1
        elif target in self.state.body(self.side):
            self.metrics['own_body_collision'] += 1
        elif target in self.state.body(self.state.enemy):
            self.metrics['rival_collision'] += 1

    def prepare_adjacent(self):
        """Find known edible neighbors of the current head."""
        self.head = self.state.head(self.side)
        self.adjacent_food_dirs = []
        self.food_kinds = {
            'numbered': self.state.food,
            'shot': self.state.shot_food,
            'star': self.state.stars,
            'tail': self.state.tail_food_a if self.side == 'A' else self.state.tail_food_b,
        }
        self.edible = frozenset().union(*self.food_kinds.values())
        if self.head is not None:
            for candidate in self.legal:
                if step(self.head, candidate) in self.edible:
                    self.adjacent_food_dirs.append(candidate)

    def observe_kinds(self):
        """Attribute adjacent food metrics to each consumable kind."""
        for kind, cells in self.food_kinds.items():
            adjacent = [
                candidate
                for candidate in self.adjacent_food_dirs
                if step(self.head, candidate) in cells
            ]
            self.record_adjacent_kind(kind, adjacent)

    def record_adjacent_kind(self, kind, adjacent):
        """Count visible and chosen neighbors of a single food kind."""
        if adjacent:
            self.metrics[f'adjacent_{kind}_food_seen'] += 1
            outcome = 'taken' if self.direction in adjacent else 'rejected'
            self.metrics[f'adjacent_{kind}_food_{outcome}'] += 1

    def observe_adjacent(self):
        """Inspect rejected adjacent food and its diagnostic pressure."""
        if not self.adjacent_food_dirs:
            return
        self.metrics['adjacent_food_seen'] += 1
        if self.direction in self.adjacent_food_dirs:
            self.metrics['adjacent_food_taken'] += 1
            return
        self.metrics['adjacent_food_rejected'] += 1
        self.observe_safe_rejection()
        if self.debug:
            self.observe_debug_pressure()

    def observe_safe_rejection(self):
        """Count at most one safe alternative per rejected adjacent-food decision."""
        for candidate in self.adjacent_food_dirs:
            apply_move(self.state, candidate, self.side)
            plan = choose_food_plan(self.state, self.side, step(self.head, candidate))
            if acceptable_plan(plan):
                self.metrics['safe_adjacent_food_rejected'] += 1
                break

    def observe_debug_pressure(self):
        """Attribute a planned immediate-food rejection to recorded search pressure."""
        plan = self.debug.get('plan') or {}
        if plan.get('our_distance') != 1 or self.direction == plan.get('first_move'):
            return
        self.record_pressure(self.debug.get('plan_move_stats') or {})

    def record_pressure(self, stats):
        """Prefer the stronger hard-pressure category when both apply."""
        if hard_food_pressure(stats):
            self.metrics['hard_adjacent_food_rejected'] += 1
        elif soft_food_pressure(stats):
            self.metrics['soft_adjacent_food_rejected'] += 1

    def observe_special(self):
        """Count multiplier, wrong-digit and wall destinations."""
        if self.head is not None:
            target_cell = step(self.head, self.direction)
            if target_cell in self.state.pickups:
                self.metrics['multiplier_pickups_us'] += 1
            elif target_cell in self.state.wrong_food():
                self.metrics['wrong_digits_us'] += 1
            elif target_cell in self.state.walls:
                self.metrics['wall_hits_us'] += 1

    def observe_food(self):
        """Update food timing from the deterministic transition."""
        after = apply_move(self.state, self.direction, self.side)
        if score_food_delta(self.state, after, self.side):
            self.metrics['foods_us'] += 1
            if self.first_food_move is None:
                self.first_food_move = self.index
            if self.last_food_turn is not None:
                self.intervals.append(self.index - self.last_food_turn)
            self.last_food_turn = self.index
        elif self.last_food_turn is None:
            self.metrics['max_turns_without_food'] = max(
                self.metrics['max_turns_without_food'], self.index + 1
            )
        else:
            self.metrics['max_turns_without_food'] = max(
                self.metrics['max_turns_without_food'], self.index - self.last_food_turn
            )

    def finish(self, turns, final, metadata, decision_debug):
        """Complete the stable report schema after the last observation."""
        self.metrics['total_turns'] = len(turns)
        self.observe_final_scores(final)
        self.metrics['first_food_move'] = (
            -1 if self.first_food_move is None else self.first_food_move
        )
        self.metrics['average_food_interval'] = (
            0 if not self.intervals else round(sum(self.intervals) / len(self.intervals), 2)
        )
        self.metrics['food_difference'] = self.metrics['foods_A'] - self.metrics['foods_B']
        self.metrics['food_per_100_moves'] = (
            0 if not turns else round(self.metrics['foods_us'] * 100 / len(turns), 2)
        )
        self.metrics['target_changes'] = self.target_changes
        self.metrics['target_abandoned'] = self.target_abandoned
        self.metrics['longest_cycle'] = self.longest_cycle
        self.metrics['self_traps'] = self.metrics['safe_adjacent_food_rejected']
        self.metrics['final'] = final or {}
        self.metrics['bot_version'] = metadata.get('version', 'unknown')
        self.metrics['bot_notes'] = metadata.get('notes', '')
        self.metrics['decision_debug_lines'] = len(decision_debug)
        return self.metrics

    def observe_final_scores(self, final):
        """Copy supplied final scores, including zero-valued scores."""
        if final:
            for side, field in (('A', 'score_1'), ('B', 'score_2')):
                value = final.get(field)
                if value is not None:
                    self.metrics[f'final_score_{side}'] = int(value)


def target_was_abandoned(previous, target, state):
    """Require the previous target to remain known before calling it abandoned."""
    return previous is not None and target != previous and previous in state.objective_cells()


def acceptable_plan(plan):
    """Recognize planner classifications accepted for an adjacent-food entry."""
    return plan is not None and plan.safety in ('SAFE', 'ACCEPTABLE_RISK')


def hard_food_pressure(stats):
    """Recognize a forced loss or immediate zero-escape rival reply."""
    return (
        stats.get('forced_zero')
        or stats.get('forced_delayed_zero')
        or stats.get('zero_escape_replies', 0) > 0
    )


def soft_food_pressure(stats):
    """Recognize constrained replies below the hard-pressure category."""
    return (
        stats.get('one_escape_replies', 0) >= 2 or stats.get('delayed_forced_zero_replies', 0) > 0
    )


def print_report(path, metrics):
    """Display the established match metrics in a stable readable order."""
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
        'wall_hits_us',
        'wrong_digits_us',
        'multiplier_pickups_us',
    ]:
        print(f"  {key}: {metrics[key]}")
    if metrics['bot_notes']:
        print(f"  bot_notes: {metrics['bot_notes']}")
    if metrics['final']:
        print(f"  final: {metrics['final']}")


def expand_log_paths(raw_paths):
    """Expand literal or wildcard log requests with the local games-directory fallback."""
    return [path for raw in raw_paths for path in resolve_log_path(raw)]


def resolve_log_path(raw):
    """Resolve one request while retaining a missing literal for a useful read error."""
    path = Path(raw)
    if any(ch in raw for ch in '*?['):
        matches = find_log_matches(raw, path)
        if matches:
            return matches
    return [local_log_path(path)]


def find_log_matches(raw, path):
    """Prefer matching working-directory logs before the games-directory fallback."""
    matches = sorted(Path('.').glob(raw))
    if not matches and path.parent == Path('.'):
        return sorted(Path('games').glob(raw))
    return matches


def local_log_path(path):
    """Find a literal log without replacing an existing working-directory file."""
    candidate = Path('games', path)
    if not path.exists() and candidate.exists():
        return candidate
    return path


def main():
    """Read selected match logs and print one analysis report per match."""
    parser = argparse.ArgumentParser()
    parser.add_argument('logs', nargs='+')
    args = parser.parse_args()
    for path in expand_log_paths(args.logs):
        print_report(path, analyze(path))


if __name__ == '__main__':
    main()
