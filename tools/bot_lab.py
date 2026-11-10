"""Full-truth offline simulations and observation-limited bot evaluation."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, field, replace
import json
import random
from itertools import product
from pathlib import Path

from tools.analyze_matches import parse_log, expand_log_paths as resolve_log_paths
from snake_bot.bot_version import BOT_VERSION
from snake_bot.bot_weights import load_active_weights, set_active_weights
from snake_bot.food_planner import (
    center_control_score,
    classify_safety,
    position_control_penalty,
)
from snake_bot.snake_brain import SnakeBrain
from snake_bot.geometry import directions_for, iter_cells
from snake_bot.game_rules import (
    ELIMINATION_SCORE_THRESHOLD,
    MAX_MOVES,
    within_blast_zone,
    bomb_respawn_candidates,
)
from snake_bot.snake_state import (
    GameState,
    apply_move,
    apply_action,
    cyclic_digit,
    in_bounds,
    legal_moves,
    parse_state,
    star_spawn_candidates,
    step,
)

DIRECTIONS = {'up', 'down', 'left', 'right'}
V7_METRICS = tuple(
    f'{metric}_{side}'
    for side in ('A', 'B')
    for metric in (
        'total_crashes',
        'first_crashes',
        'subsequent_crashes',
        'unknown_crashes',
        'crashes_border',
        'crashes_self',
        'crashes_rival',
        'crashes_wall',
        'lost_length',
        'tail_generated_for',
        'tail_collected',
        'tail_cleaned',
        'tail_points',
        'threshold_eliminations',
    )
)
MECHANICS_VALIDATION_STATUS = {
    'crash_and_tail': 'implemented according to official contract; pending live validation',
    'wrong_number_growth': 'observed in real v7 snapshots',
}


@dataclass
class SimResult:
    """Store a completed simulation's scores, event counts and termination provenance."""

    seed: int
    candidate_side: str | None
    winner: str | None
    turns: int
    score_a: int
    score_b: int
    foods_a: int
    foods_b: int
    pickups_a: int
    pickups_b: int
    wrong_digits_a: int
    wrong_digits_b: int
    wall_hits_a: int
    wall_hits_b: int
    control_moves_a: int
    control_moves_b: int
    no_legal_side: str | None
    edge_moves: int
    center_score_total: int
    diagnostics: dict[str, int] = field(default_factory=dict)
    termination_reason: str | None = None
    move_budget: int = MAX_MOVES
    remaining_moves: int | None = None


def render_board(state):
    """Render a historical board or the active player's three cube planes."""
    symbols = board_symbols(state)
    if state.levels > 1:
        return '\n\n'.join(cube_planes(state, symbols).values())
    return render_plane(state.rows, state.cols, lambda r, c: symbols.get((r, c), ' '))


def board_symbols(state):
    """Index full simulator truth once, keeping invisible cells out of protocol views."""
    symbols = {}
    layers = (
        (state.walls, '#'),
        (state.food, '*'),
        (state.bombs_large, '\U0001f4a3'),
        (state.bombs_small, '\U0001f9e8'),
        (state.shot_food, '*'),
        (state.stars, '\u2b50'),
        (state.pickups, 'X'),
        (state.tail_food_a, '\u24b6'),
        (state.tail_food_b, '\u24b7'),
    )
    for cells, symbol in layers:
        symbols.update(dict.fromkeys(cells, symbol))
    symbols.update({cell: str(digit) for cell, digit in state.food_values.items()})
    add_snake_symbols(state, symbols)
    return symbols


def add_snake_symbols(state, symbols):
    """Overlay ordered bodies with the invincibility-dependent head marker."""
    for side in ('A', 'B'):
        body = state.body(side)
        head_symbol, body_symbol = snake_symbols(side, state.invincibility(side))
        for index, cell in enumerate(reversed(body)):
            symbols[cell] = head_symbol if index == len(body) - 1 else body_symbol


def snake_symbols(side, invincible):
    """Return the visual head and body markers for a protection timer."""
    return (side.lower(), side) if invincible else (side, side.lower())


def render_plane(rows, cols, symbol_at):
    """Render one Unicode cell per column with the server's pipe borders."""
    return '\n'.join(
        '|' + ''.join(symbol_at(r, c) for c in range(cols)) + '|' for r in range(rows)
    )


def cube_planes(state, symbols=None):
    """Project full truth through the actor's head in protocol x/y/z order."""
    symbols = board_symbols(state) if symbols is None else symbols
    level, row, col = state.head(state.side)
    return {
        'board_x': render_plane(
            state.levels, state.rows, lambda layer, r: symbols.get((layer, r, col), ' ')
        ),
        'board_y': render_plane(
            state.levels, state.cols, lambda layer, c: symbols.get((layer, row, c), ' ')
        ),
        'board_z': render_plane(
            state.rows, state.cols, lambda r, c: symbols.get((level, r, c), ' ')
        ),
    }


def make_turn_data(state, game_id, turn_index):
    """Expose only the current actor's observation and public match counters."""
    data = {
        'game_id': game_id,
        'turn_token': f't_{turn_index}',
        'side': state.side,
        'board': render_board(state),
        'rows': state.rows,
        'cols': state.cols,
        'board_size': f'{state.rows}x{state.cols}',
        'score_1': state.scores.get('A', 0),
        'score_2': state.scores.get('B', 0),
        'remaining_moves': state.remaining_moves,
        'multiplier_1': state.multipliers.get('A', 1),
        'multiplier_2': state.multipliers.get('B', 1),
        'invincible_1': state.invincible_a,
        'invincible_2': state.invincible_b,
        'timeouts_1': state.timeouts_a,
        'timeouts_2': state.timeouts_b,
    }
    if state.levels > 1:
        data.update(cube_planes(state))
        data.update(
            levels=state.levels,
            head=dict(zip(('level', 'row', 'col'), state.head(state.side))),
            board_size=f'{state.levels}x{state.rows}x{state.cols}',
        )
    return data


def spawn_food(state, rng, food_count, numbered=True, pickup_count=None):
    """Replenish resources on full simulator truth, never inside blast zones."""
    state = respawn_bombs(state, rng)
    state = respawn_stars(state, rng)
    food = set(state.food)
    food_values = dict(state.food_values)
    pickups = set(state.pickups)
    walls = set(state.walls)
    food_copy_targets = dict(state.food_copy_targets)
    empties = edible_empty_cells(state)
    rng.shuffle(empties)
    next_digit = state.next_food_digit or 1
    if numbered:
        desired = [cyclic_digit(next_digit, offset) for offset in range(food_count)]
        food_copy_targets = copy_targets(state, rng, desired)
        replenish_digits(food_values, food_copy_targets, empties)
        food = target_food_cells(food_values, next_digit)
    else:
        replenish_cells(food, food_count, empties)
    replenish_cells(pickups, desired_pickups(state, pickup_count), empties)
    walls = replenished_wall(state, walls, empties, rng)
    state = replace(
        state,
        food=frozenset(food),
        food_values=food_values,
        next_food_digit=numbered_target(numbered, next_digit),
        pickups=frozenset(pickups),
        walls=frozenset(walls),
        food_copy_targets=food_copy_targets,
    )
    return replace(state, board=render_board(state))


def target_food_cells(values, target):
    """Index copies of the current ordered digit independently of other digits."""
    return {pos for pos, digit in values.items() if digit == target}


def desired_pickups(state, explicit):
    """Honor a caller's resource override or the official version quantity."""
    return resource_count(state, 2, 6) if explicit is None else explicit


def replenished_wall(state, walls, empties, rng):
    """Replace a vanished wall while keeping a surviving plate or line intact."""
    if walls:
        return walls
    return set(spawn_wall(empties, state.rows, state.cols, rng, state.levels))


def numbered_target(numbered, digit):
    """Keep legacy unordered-food simulations free of a numeric sequence."""
    return digit if numbered else None


def resource_count(state, flat, cube):
    """Select the official quantity for the state's board generation."""
    return cube if state.levels > 1 else flat


def edible_cells(state):
    """Collect every consumable, including both beneficiaries' tail food."""
    return set().union(
        state.food,
        state.food_values,
        state.pickups,
        state.tail_food_a,
        state.tail_food_b,
        state.shot_food,
        state.stars,
    )


def edible_empty_cells(state):
    """Find free full-truth cells eligible for food and pickups."""
    occupied = state.occupied() | edible_cells(state) | state.bombs_large | state.bombs_small
    return [
        cell
        for cell in iter_cells(state.rows, state.cols, state.levels)
        if cell not in occupied and not in_blast_zone(state, cell)
    ]


def copy_targets(state, rng, desired):
    """Keep each digit's sampled copy target until its window entry expires."""
    low, high = (10, 15) if state.levels > 1 else (3, 5)
    return {digit: state.food_copy_targets.get(digit, rng.randint(low, high)) for digit in desired}


def replenish_digits(values, targets, empties):
    """Populate each active digit from a shared reservoir of safe empty cells."""
    for digit, target in targets.items():
        count = sum(value == digit for value in values.values())
        for _ in range(min(target - count, len(empties))):
            values[empties.pop()] = digit


def replenish_cells(cells, target, empties):
    """Fill a resource set up to its target without reusing a cell."""
    while len(cells) < target and empties:
        cells.add(empties.pop())


def in_blast_zone(state, cell):
    """Use the shared rules to test whether a full-truth cell is unsafe for food."""
    return within_blast_zone(cell, state.bombs_large, state.bombs_small)


def respawn_bombs(state, rng):
    """Only the lab supplies RNG; core transitions retain unknown respawn types."""
    if state.remaining_moves <= 100:
        return replace(
            state,
            bombs_large=frozenset(),
            bombs_small=frozenset(),
            pending_bomb_respawns=(),
        )
    large, small = set(state.bombs_large), set(state.bombs_small)
    pending = []
    for kind in state.pending_bomb_respawns:
        candidates = available_bomb_spawns(state, kind, large | small)
        if candidates:
            (large if kind == 'large' else small).add(rng.choice(candidates))
        else:
            pending.append(kind)
    return replace(
        state,
        bombs_large=frozenset(large),
        bombs_small=frozenset(small),
        pending_bomb_respawns=tuple(pending),
    )


def available_bomb_spawns(state, kind, bombs):
    """Delegate bomb placement constraints to the shared rule implementation."""
    heads = tuple(state.head(side) for side in ('A', 'B') if state.head(side) is not None)
    edible = edible_cells(state)
    return bomb_respawn_candidates(
        state.rows,
        state.cols,
        kind,
        heads,
        occupied=state.occupied() | edible | bombs,
        edible=edible,
        levels=state.levels,
    )


def respawn_stars(state, rng):
    """Resolve only eligible unknown spawns; search never chooses random cells."""
    if protection_blocks_star(state):
        return replace(state, pending_star_spawns=0)
    missing = missing_star_count(state)
    while missing:
        candidates = star_spawn_candidates(state)
        if not candidates:
            break
        state = replace(state, stars=state.stars | {rng.choice(candidates)})
        missing -= 1
    return replace(state, pending_star_spawns=missing)


def protection_blocks_star(state):
    """Suppress ordinary stars during either snake's protection interval."""
    return not state.star_frenzy and (state.invincible_a or state.invincible_b)


def missing_star_count(state):
    """Calculate due stars using the official flat or cube frenzy quantity."""
    if state.star_frenzy:
        return max(0, resource_count(state, 10, 30) - len(state.stars))
    return min(state.pending_star_spawns, max(0, 1 - len(state.stars)))


def spawn_wall(empties, rows, cols, rng, levels=1):
    """Choose a free shrinking line or an axis-aligned square cube plate."""
    if levels > 1:
        return spawn_plate(empties, rows, cols, levels, rng)
    available = set(empties)
    lengths = range(3, min(11, max(rows, cols)) + 1, 2)
    candidates = [
        cells
        for length in lengths
        for cells in wall_lines(rows, cols, length)
        if set(cells) <= available
    ]
    return choose_wall_candidate(candidates, rng)


def choose_wall_candidate(candidates, rng):
    """Choose one existing candidate, allowing a crowded board to have none."""
    return frozenset(rng.choice(candidates)) if candidates else frozenset()


def wall_lines(rows, cols, length):
    """Enumerate flat wall candidates in historical horizontal-then-vertical order."""
    yield from horizontal_lines(rows, cols, length)
    yield from vertical_lines(rows, cols, length)


def horizontal_lines(rows, cols, length):
    """Yield each in-bounds horizontal line of a given odd length."""
    for row, start in product(range(rows), range(cols - length + 1)):
        yield tuple((row, start + offset) for offset in range(length))


def vertical_lines(rows, cols, length):
    """Yield each in-bounds vertical line of a given odd length."""
    for col, start in product(range(cols), range(rows - length + 1)):
        yield tuple((start + offset, col) for offset in range(length))


def plate_cells(center, side, axis):
    """Build a square plate centered on one fixed coordinate axis."""
    radius = side // 2
    ranges = [range(value - radius, value + radius + 1) for value in center]
    ranges[axis] = (center[axis],)
    return frozenset(product(*ranges))


def shrink_plate(cells):
    """Remove the outer ring from a fully known square plate."""
    if not cells:
        return frozenset()
    axes = tuple(zip(*cells))
    bounds = [(min(values), max(values)) for values in axes]
    return inner_plate_cells(cells, bounds)


def inner_plate_cells(cells, bounds):
    """Keep only coordinates strictly inside both varying plate axes."""
    varying = [axis for axis, (low, high) in enumerate(bounds) if low != high]
    return frozenset(cell for cell in cells if inside_plate_ring(cell, varying, bounds))


def inside_plate_ring(cell, varying, bounds):
    """Test membership in the inner square, with a singleton disappearing."""
    return bool(varying) and all(
        bounds[axis][0] < cell[axis] < bounds[axis][1] for axis in varying
    )


def spawn_plate(empties, rows, cols, levels, rng):
    """Sample free centers until a legal odd-sized plate fits in the cube."""
    available = set(empties)
    centers = list(empties)
    rng.shuffle(centers)
    for center in centers:
        plate = plate_at_center(center, available, rng)
        if plate:
            return plate
    return frozenset()


def plate_at_center(center, available, rng):
    """Try randomized orientations and sizes without constructing a candidate cube."""
    choices = list(product(range(3), range(3, 12, 2)))
    rng.shuffle(choices)
    for axis, side in choices:
        cells = plate_cells(center, side, axis)
        if cells <= available:
            return cells
    return frozenset()


def validate_move_budget(max_turns):
    """Reject budgets outside the supported global-move limit."""
    if not isinstance(max_turns, int) or not 0 <= max_turns <= MAX_MOVES:
        raise ValueError(f'max_turns must be an integer from 0 to {MAX_MOVES}')


def initial_state(
    seed, rows=20, cols=20, food_count=5, max_turns=MAX_MOVES, numbered=True, levels=1
):
    """Create complete seeded simulator truth with official resource quantities."""
    validate_move_budget(max_turns)
    rng = random.Random(seed)
    middle = rows // 2
    snakes = {
        'A': ((middle, 2), (middle, 1), (middle, 0)),
        'B': ((middle, cols - 3), (middle, cols - 2), (middle, cols - 1)),
    }
    if levels > 1:
        snakes = {
            side: tuple((levels // 2, *cell) for cell in body) for side, body in snakes.items()
        }
    state = GameState(
        rows=rows,
        cols=cols,
        board=(),
        side='A',
        enemy='B',
        snakes=snakes,
        food=frozenset(),
        scores={'A': 0, 'B': 0},
        remaining_moves=max_turns,
        reliable_tails=frozenset({'A', 'B'}),
        next_food_digit=1 if numbered else None,
        multipliers={'A': 1, 'B': 1},
        food_copy_targets={},
        crashed_a=False,
        crashed_b=False,
        levels=levels,
    )
    state = replace(
        state,
        pending_bomb_respawns=initial_bomb_types(state),
    )
    return spawn_food(state, rng, food_count, numbered=numbered)


def initial_bomb_types(state):
    """Schedule all initial bombs only before their permanent expiry."""
    if state.remaining_moves <= 100:
        return ()
    return ('large',) * resource_count(state, 2, 6) + ('small',) * resource_count(state, 5, 15)


def winner_for(state, dead_side=None):
    # dead_side is retained for callers using the old signature, not v6 rules.
    """Resolve terminal outcomes or compare scores at a shortened simulation budget."""
    if state.is_terminal():
        return state.terminal_winner()
    score_a = state.scores.get('A', 0)
    score_b = state.scores.get('B', 0)
    if score_a == score_b:
        return None
    return 'A' if score_a > score_b else 'B'


def lab_transition(state, direction, action_type='move'):
    """Apply shared core rules and collect deterministic full-truth diagnostics."""
    if direction not in directions_for(state):
        raise ValueError(f'Invalid direction: {direction!r}')
    after = (
        apply_move(state, direction, state.side)
        if action_type == 'move'
        else apply_action(state, action_type, direction, state.side)
    )
    events = action_events(state, after, action_type)
    if action_type == 'shoot' or not state.body(state.side):
        return after, events
    collect_movement_events(state, after, direction, events)
    collect_tail_generation(state, after, events)
    return after, events


def action_events(state, after, action_type):
    """Record action metadata independently of movement and crash classification."""
    events = Counter()
    side, metadata = state.side, after.last_action
    flags = {
        'star_consumed': 'stars_collected',
        'ram_cut': 'ram_cuts',
        'shot_blocked_by_invincibility': 'shots_blocked_by_invincibility',
        'bomb_triggered': 'bombs_triggered',
    }
    for flag, metric in flags.items():
        record_flag(events, side, metric, metadata.get(flag))
    if metadata.get('ram_cut'):
        events[f'ram_cut_length_{side}'] += metadata.get('cut_length', 0)
    if action_type == 'shoot':
        events.update(shoot_events(state, after))
    return events


def record_flag(events, side, metric, enabled):
    """Add an event only when present, preserving sparse diagnostic payloads."""
    if enabled:
        events[f'{metric}_{side}'] += 1


def shoot_events(state, after):
    """Count paid shots, first hits and resulting edible cut cells."""
    side = state.side
    return {
        f'shots_{side}': int(state.scores.get(side, 0) >= 5000),
        f'shot_hits_{side}': int(bool(after.last_action.get('hit'))),
        f'shot_food_generated_{side}': len(after.shot_food - state.shot_food),
    }


def was_crash(state, after):
    """Distinguish a regular stationary crash from a protected collision."""
    side = state.side
    history = after.crashed_a if side == 'A' else after.crashed_b
    return (
        not state.invincibility(side) and after.head(side) == state.head(side) and history is True
    )


def collect_movement_events(state, after, direction, events):
    """Classify the move using state transitions and exact destination coordinates."""
    side = state.side
    target = step(state.head(side), direction)
    events[f'shot_food_collected_{side}'] += int(target in state.shot_food)
    if was_crash(state, after):
        collect_crash_events(state, after, target, events)
    else:
        collect_consumption_events(state, after, target, events)
    events[f'wall_hits_{side}'] += int(target in state.walls and not state.invincibility(side))


def crash_cause(state, target):
    """Name the first collision surface in the same priority as core rules."""
    if not in_bounds(state, target):
        return 'border'
    if target in state.walls:
        return 'wall'
    if target in state.body(state.side):
        return 'self'
    return 'rival'


def collect_crash_events(state, after, target, events):
    """Record crash history and lost body length without replaying the action."""
    side = state.side
    history = state.crashed_a if side == 'A' else state.crashed_b
    kind = {False: 'first', True: 'subsequent', None: 'unknown'}[history]
    events[f'total_crashes_{side}'] += 1
    events[f'{kind}_crashes_{side}'] += 1
    events[f'crashes_{crash_cause(state, target)}_{side}'] += 1
    events[f'lost_length_{side}'] += len(state.body(side)) - len(after.body(side))


def collect_consumption_events(state, after, target, events):
    """Attribute tail, digit and pickup consumption to its actual beneficiary."""
    side = state.side
    own_tail = state.tail_food_a if side == 'A' else state.tail_food_b
    other_tail = state.tail_food_b if side == 'A' else state.tail_food_a
    collect_food_kind(state, after, target, own_tail, events)
    events[f'tail_cleaned_{side}'] += int(target in other_tail)
    events[f'pickups_{side}'] += int(target in state.pickups)


def collect_food_kind(state, after, target, own_tail, events):
    """Preserve distinct diagnostic counts for tail and numbered food."""
    side = state.side
    if target in own_tail:
        events[f'tail_collected_{side}'] += 1
        events[f'tail_points_{side}'] += after.scores[side] - state.scores[side]
    elif target in state.food:
        events[f'foods_{side}'] += 1
    elif target in state.wrong_food():
        events[f'wrong_digits_{side}'] += 1


def collect_tail_generation(state, after, events):
    """Count newly detached tail and score-threshold eliminations for each side."""
    for side in ('A', 'B'):
        before_cells = getattr(state, f'tail_food_{side.lower()}')
        after_cells = getattr(after, f'tail_food_{side.lower()}')
        events[f'tail_generated_for_{side}'] += len(after_cells - before_cells)
        if state.scores[side] >= ELIMINATION_SCORE_THRESHOLD > after.scores[side]:
            events[f'threshold_eliminations_{side}'] += 1


def observe_simulation_state(brain, state, data):
    """Feed cube bots only protocol observations, retaining the historical adapter."""
    if state.levels > 1:
        brain.observe_state(data)
        return
    game_id = str(data['game_id'])
    history = {'A': state.crashed_a, 'B': state.crashed_b}
    brain.crash_history[game_id] = dict(history)
    brain.previous_snakes[game_id] = state.snakes
    observed = brain.observe_state(data)
    # The observer infers history from sparse server snapshots. Here the local
    # transition is authoritative, including None on reconstructed scenarios.
    observed = replace(observed, crashed_a=state.crashed_a, crashed_b=state.crashed_b)
    brain.crash_history[game_id] = dict(history)
    brain.observed_states[game_id] = observed
    brain.current_states[game_id] = observed


def simulate_game(
    seed,
    max_turns=MAX_MOVES,
    rows=None,
    cols=None,
    food_count=5,
    weights_by_side=None,
    candidate_side=None,
    levels=1,
):
    """Run seeded self-play with independent brains and actor-specific observations."""
    validate_move_budget(max_turns)
    rng = random.Random(seed)
    dimensions = simulation_dimensions(rng, rows, cols, levels)
    state = simulation_initial_state(seed, dimensions, food_count, max_turns, levels)
    session = SimulationSession(state, rng, seed, food_count, weights_by_side)
    session.run(max_turns)
    return simulation_result(
        session.state, session.metrics, seed, candidate_side, session.turns, max_turns
    )


def simulation_dimensions(rng, rows, cols, levels):
    """Preserve flat seeded sizes while selecting the fixed official cube."""
    if levels > 1:
        return 20, 20
    return rows or rng.randint(20, 30), cols or rng.randint(20, 30)


def simulation_initial_state(seed, dimensions, food_count, max_turns, levels):
    """Keep the historical factory call compatible with existing tool integrations."""
    if levels > 1:
        return initial_state(seed, *dimensions, food_count, max_turns, levels=levels)
    return initial_state(seed, *dimensions, food_count, max_turns)


class SimulationSession:
    """Own one true board while exposing separate observations to each bot."""

    def __init__(self, state, rng, seed, food_count, weights_by_side):
        self.state = state
        self.rng = rng
        self.game_id = f'sim_{seed}'
        self.food_count = food_count
        self.weights_by_side = weights_by_side
        self.brains = {side: SnakeBrain(enable_debug=False) for side in ('A', 'B')}
        self.metrics = Counter()
        self.turns = 0

    def run(self, max_turns):
        """Stop on the simulation budget or a terminal shared-rule state."""
        for turn_index in range(max_turns):
            if self.state.is_terminal():
                break
            self.play_turn(turn_index)
            if self.state.is_terminal():
                break
            self.state = spawn_food(self.state, self.rng, self.food_count)

    def play_turn(self, turn_index):
        """Ask one bot, apply core rules once, then commit the observed proposal."""
        state, side = self.state, self.state.side
        self.select_weights(side)
        data = make_turn_data(state, self.game_id, turn_index)
        brain = self.brains[side]
        observe_simulation_state(brain, state, data)
        action = simulation_action(brain, data)
        direction, action_type = action['direction'], action['action_type']
        if direction not in directions_for(state):
            raise ValueError(f'Invalid direction: {direction!r}')
        collect_decision_events(state, brain, self.game_id, action, self.metrics)
        self.state, events = lab_transition(state, direction, action_type)
        self.metrics.update(events)
        self.turns += 1
        self.metrics['center_score_total'] += center_control_score(
            self.state, self.state.head(side)
        )
        commit_simulation_action(brain, data, action)

    def select_weights(self, side):
        """Apply only the explicit benchmark configuration for the active side."""
        if self.weights_by_side and side in self.weights_by_side:
            set_active_weights(self.weights_by_side[side])


def simulation_action(brain, data):
    """Accept current action records and preserve the legacy move-only adapter."""
    action = brain.choose_action(data)
    if action is None:
        raise ValueError('Invalid direction: None')
    if not isinstance(action, dict):
        return {'action_type': 'move', 'direction': brain.choose_move(data)}
    return action


def collect_decision_events(state, brain, game_id, action, metrics):
    """Count decision reasons and edge preference separately from game events."""
    debug = brain.decision_debug(game_id) or {}
    if debug.get('reason') == 'advantage_control':
        metrics[f'control_moves_{state.side}'] += 1
    if (
        action['action_type'] == 'move'
        and position_control_penalty(state, state.side, action['direction']) > 0
    ):
        metrics['edge_moves'] += 1


def commit_simulation_action(brain, data, action):
    """Advance only the active bot's private prediction and observation history."""
    if action['action_type'] == 'move':
        brain.commit_move(data, action['direction'])
    else:
        brain.commit_action(data, action)


def simulation_termination(state, turns, max_turns):
    """Describe why a simulation stopped without claiming a short budget is official."""
    if any(score < ELIMINATION_SCORE_THRESHOLD for score in state.scores.values()):
        return 'score_elimination'
    if turns == max_turns and max_turns < MAX_MOVES:
        return 'simulation_budget'
    return 'movement_limit'


def simulation_result(state, metrics, seed, candidate_side, turns, max_turns):
    """Build the stable benchmark result schema from accumulated turn metrics."""
    termination_reason = simulation_termination(state, turns, max_turns)
    return SimResult(
        seed=seed,
        candidate_side=candidate_side,
        winner=(state.terminal_winner() if state.is_terminal() else winner_for(state)),
        turns=turns,
        score_a=state.scores.get('A', 0),
        score_b=state.scores.get('B', 0),
        foods_a=metrics['foods_A'],
        foods_b=metrics['foods_B'],
        pickups_a=metrics['pickups_A'],
        pickups_b=metrics['pickups_B'],
        wrong_digits_a=metrics['wrong_digits_A'],
        wrong_digits_b=metrics['wrong_digits_B'],
        wall_hits_a=metrics['wall_hits_A'],
        wall_hits_b=metrics['wall_hits_B'],
        control_moves_a=metrics['control_moves_A'],
        control_moves_b=metrics['control_moves_B'],
        no_legal_side=None,
        edge_moves=metrics['edge_moves'],
        center_score_total=metrics['center_score_total'],
        diagnostics=dict(metrics),
        termination_reason=termination_reason,
        move_budget=max_turns,
        remaining_moves=state.remaining_moves,
    )


SUMMARY_SIDE_FIELDS = (
    'foods',
    'pickups',
    'wrong_digits',
    'wall_hits',
    'control_moves',
)
SUMMARY_CANDIDATE_FIELDS = ('foods', 'wall_hits', 'control_moves')


def summarize_simulations(results):
    """Combine game outcomes and candidate-relative metrics without changing weights."""
    totals = simulation_side_totals(results)
    candidates = [row for row in results if row.candidate_side in ('A', 'B')]
    return {
        'bot_version': BOT_VERSION,
        'games': len(results),
        **simulation_score_totals(results),
        **simulation_outcomes(results),
        **totals,
        **simulation_turn_rates(results, totals),
        **candidate_outcomes(candidates),
        **candidate_totals(candidates),
        **simulation_diagnostics(results),
    }


def simulation_total(results, attribute):
    """Sum one stable result field across completed simulations."""
    return sum(getattr(row, attribute) for row in results)


def rounded_ratio(numerator, denominator):
    """Report a two-decimal rate and a numeric zero for an empty denominator."""
    return round(numerator / denominator, 2) if denominator else 0


def simulation_side_totals(results):
    """Collect activity totals separately for both fixed player sides."""
    return {
        f'{name}_{side}': simulation_total(results, f'{name}_{side.lower()}')
        for name in SUMMARY_SIDE_FIELDS
        for side in ('A', 'B')
    }


def simulation_score_totals(results):
    """Report mean turns and score differences across completed games."""
    score_a = simulation_total(results, 'score_a')
    score_b = simulation_total(results, 'score_b')
    games = len(results)
    return {
        'avg_turns': rounded_ratio(simulation_total(results, 'turns'), games),
        'avg_score_A': rounded_ratio(score_a, games),
        'avg_score_B': rounded_ratio(score_b, games),
        'avg_total_game_score': rounded_ratio(score_a + score_b, games),
        'avg_score_diff_A_minus_B': rounded_ratio(score_a - score_b, games),
    }


def simulation_outcomes(results):
    """Count winners, draws and legacy no-legal-move outcomes."""
    wins = Counter(row.winner or 'draw' for row in results)
    deaths = Counter(row.no_legal_side for row in results if row.no_legal_side)
    return {
        'wins_A': wins['A'],
        'wins_B': wins['B'],
        'draws': wins['draw'],
        'deaths_A': deaths['A'],
        'deaths_B': deaths['B'],
    }


def simulation_turn_rates(results, totals):
    """Normalize food and positional activity by global moves played."""
    turns = simulation_total(results, 'turns')
    foods = totals['foods_A'] + totals['foods_B']
    return {
        'food_per_100_turns': rounded_ratio(foods * 100, turns),
        'edge_moves_per_100_turns': rounded_ratio(
            simulation_total(results, 'edge_moves') * 100, turns
        ),
        'avg_center_score': rounded_ratio(simulation_total(results, 'center_score_total'), turns),
    }


def candidate_outcomes(results):
    """Orient win and death counts to each result's candidate assignment."""
    wins, losses = participant_events(results, 'winner')
    deaths, opponent_deaths = participant_events(results, 'no_legal_side')
    return {
        'candidate_games': len(results),
        'candidate_wins': wins,
        'candidate_losses': losses,
        'candidate_deaths': deaths,
        'opponent_deaths': opponent_deaths,
    }


def participant_events(results, attribute):
    """Count an outcome for the candidate and for its opponent."""
    events = Counter(
        getattr(row, attribute) == row.candidate_side
        for row in results
        if getattr(row, attribute) in ('A', 'B')
    )
    return events[True], events[False]


def candidate_totals(results):
    """Report candidate-relative scores alongside resource activity."""
    own, other = participant_totals(results, 'score')
    games = len(results)
    return {
        'avg_candidate_score': rounded_ratio(own, games),
        'avg_opponent_score': rounded_ratio(other, games),
        'avg_candidate_score_diff': rounded_ratio(own - other, games),
        **candidate_activity(results),
    }


def candidate_activity(results):
    """Collect resource events using the candidate assignment in each game."""
    totals = {}
    for metric in SUMMARY_CANDIDATE_FIELDS:
        own, other = participant_totals(results, metric)
        totals[f'candidate_{metric}'] = own
        totals[f'opponent_{metric}'] = other
    return totals


def participant_totals(results, field):
    """Orient numeric side fields before summing candidate and opponent totals."""
    totals = [0, 0]
    for row in results:
        a = getattr(row, f'{field}_a')
        b = getattr(row, f'{field}_b')
        own, other = (a, b) if row.candidate_side == 'A' else (b, a)
        totals[0] += own
        totals[1] += other
    return totals


def simulation_diagnostics(results):
    """Preserve termination provenance and aggregate rule-event counters."""
    return {
        **diagnostic_totals(results),
        'termination_reasons': dict(
            Counter(row.termination_reason or 'unspecified' for row in results)
        ),
        'move_budgets': dict(Counter(row.move_budget for row in results)),
        'mechanics_validation_status': dict(MECHANICS_VALIDATION_STATUS),
    }


def diagnostic_totals(results):
    """Sum the stable crash-and-tail diagnostic schema."""
    return {key: sum(row.diagnostics.get(key, 0) for row in results) for key in V7_METRICS}


def iter_log_entries(paths):
    """Yield chronological turn snapshots from each selected saved match."""
    for path in paths:
        turns, _, _, _, _ = parse_log(Path(path))
        for index, data in enumerate(turns):
            yield path, index, data


def select_log_entries(paths, max_positions=None):
    """Sample log positions at a fixed stride when an explicit limit is supplied."""
    entries = list(iter_log_entries(paths))
    if not max_positions or max_positions >= len(entries):
        return entries, False
    stride = max(1, len(entries) // max_positions)
    selected = entries[::stride][:max_positions]
    return selected, True


def evaluate_log_decisions(paths, max_positions=None):
    """Re-evaluate saved observations with isolated or chronological bot memory."""
    entries, sampled = select_log_entries(paths, max_positions)
    sequential_brain = SnakeBrain(enable_debug=False)
    metrics, by_reason = Counter(), Counter()
    for _, _, data in entries:
        brain = SnakeBrain(enable_debug=False) if sampled else sequential_brain
        evaluate_log_position(brain, data, sampled, metrics, by_reason)
    metrics['bot_version'] = BOT_VERSION
    metrics['sampled_positions'] = sampled
    metrics['reasons'] = dict(by_reason)
    return metrics


def evaluate_log_position(brain, data, sampled, metrics, by_reason):
    """Classify one saved decision without granting the bot hidden simulator truth."""
    state = parse_state(data)
    legal = legal_moves(state, state.side)
    metrics['positions'] += 1
    if not legal:
        metrics['no_legal_positions'] += 1
        return
    direction = brain.safe_direction(data, brain.choose_move(data))
    debug = brain.decision_debug(data.get('game_id', 'default')) or {}
    by_reason[debug.get('reason', 'unknown')] += 1
    if not valid_log_direction(state, direction, legal, metrics):
        return
    collect_log_move(state, direction, metrics)
    if not sampled:
        brain.commit_move(data, direction)


def valid_log_direction(state, direction, legal, metrics):
    """Keep invalid protocol directions separate from unsafe legal proposals."""
    if direction not in directions_for(state):
        metrics['invalid_direction'] += 1
        return False
    if direction not in legal:
        metrics['illegal_after_repair'] += 1
        return False
    return True


def collect_log_move(state, direction, metrics):
    """Evaluate the observed move's safety and immediate resource destination."""
    side = state.side
    after = apply_move(state, direction, side)
    safety, _, _, _ = classify_safety(after, side)
    metrics[f'safety_{safety}'] += 1
    target = step(state.head(side), direction)
    flags = {
        'food_taken_now': target in state.food | state.shot_food,
        'shot_food_taken_now': target in state.shot_food,
        'wall_hit': target in state.walls,
        'edge_or_corner_move': position_control_penalty(state, side, direction) > 0,
    }
    for metric, present in flags.items():
        if present:
            metrics[metric] += 1


def expand_log_paths(raw_paths):
    """Resolve local and games-directory logs using the shared analysis contract."""
    return resolve_log_paths(raw_paths)


def print_json(payload):
    """Write a deterministic human-readable JSON report to standard output."""
    print(json.dumps(payload, indent=2, sort_keys=True))


def main():
    """Parse the simulator or saved-log command and display its requested report."""
    parser = argparse.ArgumentParser(description='Offline lab for the Snake bot.')
    sub = parser.add_subparsers(dest='command', required=True)

    sim = sub.add_parser('simulate', help='run deterministic self-play simulations')
    sim.add_argument('--games', type=int, default=50)
    sim.add_argument('--seed', type=int, default=1)
    sim.add_argument('--turns', type=int, default=MAX_MOVES)
    sim.add_argument(
        '--rows',
        type=int,
        default=0,
        help='0 chooses a random size from 20 to 30',
    )
    sim.add_argument(
        '--cols',
        type=int,
        default=0,
        help='0 chooses a random size from 20 to 30',
    )
    sim.add_argument('--food', type=int, default=5)
    sim.add_argument(
        '--levels',
        type=int,
        choices=(1, 20),
        default=None,
        help='20 selects the official cube; explicit flat dimensions retain legacy mode',
    )
    sim.add_argument('--json', action='store_true')

    logs = sub.add_parser('logs', help='reevaluate historical log positions with the current bot')
    logs.add_argument('paths', nargs='+')
    logs.add_argument('--max-positions', type=int, default=0)
    logs.add_argument('--json', action='store_true')

    args = parser.parse_args()
    load_active_weights()
    if args.command == 'simulate':
        results = [simulate_cli_game(args, offset) for offset in range(args.games)]
        summary = summarize_simulations(results)
        display_simulations(summary, args.json)
    elif args.command == 'logs':
        summary = evaluate_log_decisions(expand_log_paths(args.paths), args.max_positions or None)
        display_summary(summary, args.json)


def simulate_cli_game(args, offset):
    """Default new CLI runs to the cube while preserving explicit historical dimensions."""
    parameters = (args.seed + offset, args.turns, args.rows, args.cols, args.food)
    if legacy_cli_dimensions(args):
        return simulate_game(*parameters)
    return simulate_game(*parameters, levels=20)


def legacy_cli_dimensions(args):
    """Recognize requested flat mode and existing explicit row/column commands."""
    return args.levels == 1 or (args.levels is None and (args.rows or args.cols))


def display_simulations(summary, as_json):
    """Select JSON or the established human-readable self-play report."""
    if as_json:
        print_json(summary)
    else:
        print_simulation_summary(summary)


def print_simulation_summary(summary):
    """Print the stable simulation metrics in their existing display order."""
    print(f"bot_version: {summary['bot_version']}")
    print(f"games: {summary['games']}")
    print(f"avg_turns: {summary['avg_turns']}")
    print((f"avg_score_A: {summary['avg_score_A']}  avg_score_B: " f"{summary['avg_score_B']}"))
    print(
        (
            f"wins_A: {summary['wins_A']}  wins_B: "
            f"{summary['wins_B']}  draws: {summary['draws']}"
        )
    )
    print((f"deaths_A: {summary['deaths_A']}  deaths_B: " f"{summary['deaths_B']}"))
    print(f"foods_A: {summary['foods_A']}  foods_B: {summary['foods_B']}")
    print((f"pickups_A: {summary['pickups_A']}  pickups_B: " f"{summary['pickups_B']}"))
    print(
        (
            f"wrong_digits_A: {summary['wrong_digits_A']}  "
            f"wrong_digits_B: {summary['wrong_digits_B']}"
        )
    )
    print((f"wall_hits_A: {summary['wall_hits_A']}  wall_hits_B: " f"{summary['wall_hits_B']}"))
    print(
        (
            f"control_moves_A: {summary['control_moves_A']}  "
            f"control_moves_B: {summary['control_moves_B']}"
        )
    )
    print(f"food_per_100_turns: {summary['food_per_100_turns']}")
    print((f"edge_moves_per_100_turns: " f"{summary['edge_moves_per_100_turns']}"))
    print(f"avg_center_score: {summary['avg_center_score']}")
    for key in V7_METRICS:
        print(f'{key}: {summary[key]}')
    for key in (
        'termination_reasons',
        'move_budgets',
        'mechanics_validation_status',
    ):
        print(f'{key}: {summary[key]}')


def display_summary(summary, as_json):
    """Print a generic report without changing the JSON schema."""
    if as_json:
        print_json(summary)
    else:
        for key, value in summary.items():
            print(f'{key}: {value}')


if __name__ == '__main__':
    main()
