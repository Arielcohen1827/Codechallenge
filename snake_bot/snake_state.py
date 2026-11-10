from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field, replace
from snake_bot import geometry
from snake_bot.geometry import directions_for
from snake_bot.observation import CubeObservation, KnowledgeState, observed_body
from snake_bot import game_rules as _rules
from snake_bot.game_rules import (
    CRASH_KEEP_LENGTH,
    ELIMINATION_SCORE_THRESHOLD,
    MAX_MOVES,
    FOOD_SCORE,
    MULTIPLIER_PICKUP_SCORE,
    WRONG_FOOD_PENALTY,
    cyclic_digit,
    find_next_food_digit,
    food_points,
    score_after_crash,
    tail_food_reward,
    BOMB_EXPIRY_MOVES,
    within_blast_zone,
    score_after_bomb,
    shot_food_reward,
    SHOOT_COST,
    INVALID_ACTION_PENALTY,
    first_shot_hit,
    bombs_active_on_turn,
    blast_zone_cells,
    STAR_TURNS,
    STAR_RESPAWN_ROUNDS,
    STAR_FRENZY_COUNT,
    MAX_TIMEOUTS,
    star_spawn_candidates as eligible_star_cells,
)

CRASH_PENALTY = _rules.CRASH_PENALTY
RIVAL_CRASH_REWARD = _rules.RIVAL_CRASH_REWARD

DIRS = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1),
}

NORMAL_SCORE = 1
WALL_HIT_PENALTY = -500
NEVER_RELEASE = 10_000


@dataclass(frozen=True)
class GameState:
    """Authoritative rule state with explicit partial-observation metadata for live games."""

    rows: int
    cols: int
    board: tuple[str, ...]
    side: str
    enemy: str
    snakes: dict[str, tuple[tuple[int, int], ...]]
    food: frozenset[tuple[int, int]]
    scores: dict[str, int]
    remaining_moves: int
    reliable_tails: frozenset[str] = frozenset()
    food_values: dict[tuple[int, int], int] = field(default_factory=dict)
    next_food_digit: int | None = None
    pickups: frozenset[tuple[int, int]] = frozenset()
    multipliers: dict[str, int] = field(default_factory=lambda: {'A': 1, 'B': 1})
    walls: frozenset[tuple[int, int]] = frozenset()
    food_copy_targets: dict[int, int] = field(default_factory=dict)
    tail_food_a: frozenset[tuple[int, int]] = frozenset()
    tail_food_b: frozenset[tuple[int, int]] = frozenset()
    crashed_a: bool | None = None
    crashed_b: bool | None = None
    bombs_large: frozenset[tuple[int, int]] = frozenset()
    bombs_small: frozenset[tuple[int, int]] = frozenset()
    shot_food: frozenset[tuple[int, int]] = frozenset()
    pending_bomb_respawns: tuple[str, ...] = ()
    last_action: dict = field(default_factory=dict, compare=False)
    stars: frozenset[tuple[int, int]] = frozenset()
    invincible_a: int = 0
    invincible_b: int = 0
    star_spawn_due_round: float | None = STAR_RESPAWN_ROUNDS
    pending_star_spawns: int = 0
    timeouts_a: int = 0
    timeouts_b: int = 0
    levels: int = 1
    partial: bool = False
    observation: object = field(default=None, compare=False)
    knowledge: object = field(default=None, compare=False)
    observed_snake_cells: dict = field(default_factory=dict)

    def invincibility(self, side):
        """Read the requested players remaining protected turns."""
        return self.invincible_a if side == 'A' else self.invincible_b

    @property
    def star_frenzy(self):
        """Whether the global movement clock has entered the final star phase."""
        return self.remaining_moves <= BOMB_EXPIRY_MOVES

    def terminal_winner(self):
        """Return the winning side only when a terminal condition determines it."""
        early = self.early_winner()
        if early is not None:
            return early
        if self.remaining_moves <= 0:
            return score_winner(self.scores)
        return None

    def early_winner(self):
        """Evaluate timeout and score eliminations before the movement limit."""
        if self.timeouts_a >= MAX_TIMEOUTS:
            return 'B'
        if self.timeouts_b >= MAX_TIMEOUTS:
            return 'A'
        return eliminated_opponent(self.scores)

    def is_terminal(self):
        """Check timeout, movement-budget and score-elimination endings."""
        return (
            max(self.timeouts_a, self.timeouts_b) >= MAX_TIMEOUTS
            or self.remaining_moves <= 0
            or any(score < ELIMINATION_SCORE_THRESHOLD for score in self.scores.values())
        )

    def head(self, side):
        """Return the confirmed first body coordinate, or None when not visible."""
        snake = self.snakes.get(side, ())
        return snake[0] if snake else None

    def body(self, side):
        """Return the ordered known body prefix for the requested side."""
        return self.snakes.get(side, ())

    def occupied(self):
        """Collect certain wall and snake cells while excluding uncertain memories."""
        cells = set(self.walls)
        for snake in self.snakes.values():
            cells.update(snake)
        for visible in self.observed_snake_cells.values():
            cells.update(visible)
        return cells

    def wrong_food(self):
        """Return numbered food cells whose target status is not confirmed."""
        return frozenset(set(self.food_values) - set(self.food))

    def objective_cells(self):
        """Collect currently observed food, pickups and stars for planning."""
        return frozenset(
            set(self.food) | set(self.pickups) | set(self.shot_food) | set(self.stars)
        )

    def food_reward(self, side, pos):
        """Compute numbered or ordinary food points using the actors multiplier."""
        digit = self.food_values.get(pos)
        return food_points(digit, self.multipliers.get(side, 1))

    def numbered_sequence(self, limit=5):
        """Choose the nearest currently available copy in each ordered digit group."""
        if self.next_food_digit is None or not self.food_values:
            return ()
        head = self.head(self.side)
        ordered = []
        for digit, positions in self.numbered_groups(limit=limit):
            if head is None:
                pos = positions[0]
            else:
                pos = min(positions, key=lambda cell: (manhattan(head, cell), cell))
            ordered.append((digit, pos))
        return tuple(ordered)

    def numbered_groups(self, limit=5):
        """Return each ordered digit with every currently available copy."""
        if self.next_food_digit is None or not self.food_values:
            return ()
        positions_by_digit = {}
        for pos, digit in self.food_values.items():
            positions_by_digit.setdefault(digit, []).append(pos)
        ordered = ordered_digit_groups(positions_by_digit, self.next_food_digit)
        return tuple(ordered[: max(1, limit)])


def score_winner(scores):
    """Resolve a completed movement budget, returning None on equal scores."""
    a, b = scores.get('A', 0), scores.get('B', 0)
    return 'A' if a > b else 'B' if b > a else None


def eliminated_opponent(scores):
    """Return the opponent of the first side below the elimination threshold."""
    for side in ('A', 'B'):
        if scores.get(side, 0) < ELIMINATION_SCORE_THRESHOLD:
            return 'B' if side == 'A' else 'A'
    return None


def ordered_digit_groups(positions_by_digit, first):
    """Group available copies in cyclic order while skipping absent digits."""
    digits = (cyclic_digit(first, offset) for offset in range(9))
    return [
        (digit, tuple(sorted(positions_by_digit[digit])))
        for digit in digits
        if digit in positions_by_digit
    ]


def parse_board(board_text, rows=None, cols=None):
    """Parse a complete historical board and normalize its requested dimensions."""
    cleaned = clean_board_rows(board_text)
    rows = len(cleaned) if rows is None else rows
    cols = board_width(cleaned) if cols is None else cols
    return sized_board(cleaned, rows, cols), rows, cols


def clean_board_rows(text):
    """Remove display borders without splitting Unicode cell symbols."""
    return [
        strip_board_border(line.rstrip('\n')) for line in text.splitlines() if line.strip('\n')
    ]


def strip_board_border(line):
    """Remove optional leading and trailing row delimiters."""
    if line.startswith('|'):
        line = line[1:]
    if line.endswith('|'):
        line = line[:-1]
    return line


def board_width(rows):
    """Find the widest available row, including an empty-board fallback."""
    return max(map(len, rows), default=0)


def sized_board(cleaned, rows, cols):
    """Pad or truncate a complete historical board to the requested geometry."""
    board = tuple(line.ljust(cols)[:cols] for line in cleaned[:rows])
    return board + (' ' * cols,) * max(0, rows - len(board))


def parse_state(data, previous_snakes=None, crash_history=None, knowledge=None):
    """Parse a server snapshot using the caller's confirmed crash history."""
    if cube_payload(data):
        return parse_cube_state(data, previous_snakes, crash_history, knowledge)
    return parse_flat_state(data, previous_snakes, crash_history)


def cube_payload(data):
    """Identify cube protocol from its dimension field or three-axis size."""
    return bool(data.get('levels')) or len(str(data.get('board_size', '')).split('x')) == 3


def parse_flat_state(data, previous_snakes=None, crash_history=None):
    """Preserve complete-observation parsing for historical two-dimensional games."""
    raw_rows, raw_cols = snapshot_board_size(data)
    board, rows, cols = parse_board(data.get('board', ''), raw_rows, raw_cols)
    side = normalize_side(data.get('side'), board)
    invincible = snapshot_player_values(data, 'invincible', minimum=0)
    snakes, reliable = observed_snakes(board, invincible, previous_snakes)
    moves = int(data.get('remaining_moves', MAX_MOVES) or 0)
    crash_history = crash_history or {}
    return GameState(
        rows=rows,
        cols=cols,
        board=board,
        side=side,
        enemy='B' if side == 'A' else 'A',
        snakes=snakes,
        scores=snapshot_player_values(data, 'score'),
        remaining_moves=moves,
        reliable_tails=frozenset(reliable),
        multipliers=snapshot_player_values(data, 'multiplier', minimum=1),
        crashed_a=crash_history.get('A'),
        crashed_b=crash_history.get('B'),
        **snapshot_food(board),
        **snapshot_objects(board, moves, invincible),
        **snapshot_counters(data, invincible),
    )


def parse_cube_state(data, previous_snakes=None, crash_history=None, knowledge=None):
    """Build a partial state from visible planes without asserting hidden contents."""
    observation = CubeObservation.from_turn_data(data)
    side = str(data.get('side', 'A')).upper()
    invincible = snapshot_player_values(data, 'invincible', minimum=0)
    snakes = cube_snakes(observation, side, invincible, previous_snakes)
    moves = int(data.get('remaining_moves', MAX_MOVES))
    known = (knowledge or KnowledgeState()).reconcile(observation, MAX_MOVES - moves)
    known = enemy_knowledge(known, snakes, side, MAX_MOVES - moves)
    history = crash_history or {}
    return GameState(
        rows=observation.rows,
        cols=observation.cols,
        levels=observation.levels,
        board=observation.planes['board_z'],
        side=side,
        enemy='B' if side == 'A' else 'A',
        snakes=snakes,
        scores=snapshot_player_values(data, 'score'),
        remaining_moves=moves,
        multipliers=snapshot_player_values(data, 'multiplier', minimum=1),
        crashed_a=history.get('A'),
        crashed_b=history.get('B'),
        invincible_a=invincible['A'],
        invincible_b=invincible['B'],
        partial=True,
        observation=observation,
        knowledge=known,
        observed_snake_cells={
            name: observation.positions((name, name.lower())) for name in ('A', 'B')
        },
        **cube_objects(observation, moves),
        **cube_food(observation, moves),
    )


def cube_snakes(observation, side, invincible, previous):
    """Decode confirmed body prefixes using the authoritative local head."""
    previous = previous or {}
    return {
        name: observed_body(
            observation,
            name,
            invincible[name],
            previous.get(name, ()),
            observation.head if name == side else None,
        )
        for name in ('A', 'B')
    }


def enemy_knowledge(known, snakes, side, turn):
    """Timestamp an actually visible rival head without promoting a remembered one."""
    rival = snakes['B' if side == 'A' else 'A']
    if rival:
        return replace(known, last_enemy_head=rival[0], last_enemy_turn=turn)
    return known


def cube_objects(observation, moves):
    """Extract observed objects while enforcing deterministic bomb expiry."""
    positions = observation.positions
    bombs = moves > BOMB_EXPIRY_MOVES
    return dict(
        pickups=positions('X'),
        walls=positions('#'),
        bombs_large=positions('💣') if bombs else frozenset(),
        bombs_small=positions('🧨') if bombs else frozenset(),
        tail_food_a=positions('Ⓐ'),
        tail_food_b=positions('Ⓑ'),
        shot_food=positions('*'),
        stars=positions('⭐'),
    )


def cube_food(observation, moves):
    """Infer an ordered target only when visible digits determine the full window."""
    values = {
        cell: int(symbol) for cell, symbol in observation.cells.items() if symbol in '123456789'
    }
    target = certain_food_target(set(values.values()), moves)
    return dict(
        food_values=values,
        next_food_digit=target,
        food=frozenset(cell for cell, digit in values.items() if digit == target),
    )


def certain_food_target(digits, moves):
    """Reject ambiguous partial windows instead of mistaking absence for evidence."""
    if moves == MAX_MOVES:
        return 1
    candidates = [start for start in range(1, 10) if digits <= food_window(start)]
    return candidates[0] if len(candidates) == 1 else None


def food_window(start):
    """Return the five consecutive active digits for one possible target."""
    return {cyclic_digit(start, offset) for offset in range(5)}


def snapshot_board_size(data):
    """Resolve historical row and column dimensions from protocol fields."""
    rows, cols = data.get('rows'), data.get('cols')
    if rows is not None and cols is not None:
        return rows, cols
    if not data.get('board_size'):
        return rows, cols
    return fill_missing_board_size(rows, cols, data['board_size'])


def fill_missing_board_size(rows, cols, size):
    """Fill omitted dimensions from a valid two-axis size string."""
    try:
        size_rows, size_cols = str(size).lower().split('x', 1)
        if rows is None:
            rows = int(size_rows.strip())
        if cols is None:
            cols = int(size_cols.strip())
    except (TypeError, ValueError):
        pass
    return rows, cols


def snapshot_player_values(data, field, minimum=None):
    """Read per-player numeric fields with an optional minimum."""
    values = {
        side: int(data.get(f'{field}_{index}', data.get(f'{field}_{side}', 0)) or 0)
        for side, index in (('A', 1), ('B', 2))
    }
    if minimum is None:
        return values
    return {side: max(minimum, value) for side, value in values.items()}


def observed_snakes(board, invincible, previous):
    """Reconstruct complete historical bodies and reconcile confirmed ordering."""
    snakes = {side: reconstruct_snake(board, side, invincible[side]) for side in ('A', 'B')}
    reliable = set()
    if previous:
        for side in ('A', 'B'):
            tracked = track_snake(previous.get(side, ()), board, side, invincible[side])
            if tracked is not None:
                snakes[side] = tracked
                reliable.add(side)
    return snakes, reliable


def snapshot_food(board):
    """Extract historical food copies and their cyclic target."""
    values = {
        position: int(symbol) for position, symbol in board_cells(board) if symbol in '123456789'
    }
    next_digit = find_next_food_digit(values.values())
    return {
        'food_values': values,
        'food_copy_targets': dict(sorted(Counter(values.values()).items())),
        'next_food_digit': next_digit,
        'food': positions_with_digit(values, next_digit),
    }


def positions_with_digit(values, digit):
    """Select every observed copy of the requested numbered food."""
    return frozenset(position for position, value in values.items() if value == digit)


def board_cells(board):
    """Yield coordinates and Unicode symbols from a complete flat board."""
    return (((row, col), char) for row, line in enumerate(board) for col, char in enumerate(line))


def cells_with_symbol(board, symbol):
    """Collect flat-board coordinates matching one cell symbol."""
    return frozenset(cell for cell, char in board_cells(board) if char == symbol)


def snapshot_objects(board, moves, invincible):
    """Read pickups, hazards and special food from a complete historical board."""
    stars = cells_with_symbol(board, '⭐')
    return {
        'pickups': cells_with_symbol(board, 'X'),
        'walls': cells_with_symbol(board, '#'),
        'tail_food_a': cells_with_symbol(board, 'Ⓐ'),
        'tail_food_b': cells_with_symbol(board, 'Ⓑ'),
        'shot_food': cells_with_symbol(board, '*'),
        'stars': stars,
        'star_spawn_due_round': initial_star_round(moves, invincible),
        **snapshot_bombs_and_frenzy(board, moves, stars),
    }


def snapshot_bombs_and_frenzy(board, moves, stars):
    """Enforce bomb disappearance and frenzy metadata at the final phase."""
    if moves > BOMB_EXPIRY_MOVES:
        return {
            'bombs_large': cells_with_symbol(board, '💣'),
            'bombs_small': cells_with_symbol(board, '🧨'),
        }
    return {'pending_star_spawns': max(0, STAR_FRENZY_COUNT - len(stars))}


def initial_star_round(moves, invincible):
    """Resolve whether a normal star deadline can still be known."""
    if moves < MAX_MOVES - 2 * STAR_RESPAWN_ROUNDS:
        return None  # The earlier expiry round may not be known.
    if any(invincible.values()):
        return None
    return STAR_RESPAWN_ROUNDS


def snapshot_counters(data, invincible):
    """Read protection and timeout counters from authoritative fields."""
    timeouts = snapshot_player_values(data, 'timeouts', minimum=0)
    return {
        'invincible_a': invincible['A'],
        'invincible_b': invincible['B'],
        'timeouts_a': timeouts['A'],
        'timeouts_b': timeouts['B'],
    }


def normalize_side(raw_side, board):
    """Resolve the active player while retaining historical protocol fallbacks."""
    side = str(raw_side or 'A').upper()
    if side in ('A', 'B'):
        return side
    if board_contains(board, 'A'):
        return 'A'
    return 'B' if board_contains(board, 'B') else 'A'


def board_contains(board, marker):
    """Check whether a complete board contains the requested marker."""
    return any(marker in row for row in board)


def reconstruct_snake(board, side, invincible=0):
    """Order a complete historical body using deterministic adjacency ties."""
    head_char = side.lower() if invincible else side
    heads = [cell for cell, char in board_cells(board) if char == head_char]
    if not heads:
        return ()
    return order_flat_body(heads[-1], snake_cells(board, side), board)


def order_flat_body(head, cells, board):
    """Walk adjacent body cells then preserve any disconnected legacy fragments."""
    ordered = [head]
    unused = set(cells) - {head}
    cols = board_width(board)
    while unused:
        candidates = flat_body_candidates(ordered[-1], unused, len(board), cols)
        if not candidates:
            break
        following = candidates[0]
        ordered.append(following)
        unused.remove(following)
    return tuple(ordered + sorted(unused))


def flat_body_candidates(current, unused, rows, cols):
    """Keep low-degree neighbors first to retain the historical reconstruction."""
    candidates = [cell for cell in neighbors(current, rows, cols) if cell in unused]
    return sorted(candidates, key=lambda cell: body_degree(cell, unused, rows, cols))


def track_snake(previous, board, side, invincible=0):
    """Recover a complete body only when a known transition exactly matches it."""
    cells = snake_cells(board, side)
    head = snake_head(board, side, invincible)
    if head is None or not cells:
        return ()
    if not previous or not previous[0]:
        return None
    return tracked_body_transition(previous, head, cells)


def tracked_body_transition(previous, head, cells):
    """Compare stationary cuts before testing an adjacent move or growth."""
    if head == previous[0]:
        candidates = (
            tuple(previous),
            tuple(previous[: len(cells)]),
            tuple(previous[:CRASH_KEEP_LENGTH]),
        )
        return matching_body(candidates, cells)
    if manhattan(head, previous[0]) != 1:
        return None
    return matching_body(((head,) + tuple(previous[:-1]), (head,) + tuple(previous)), cells)


def matching_body(candidates, cells):
    """Select the first ordered candidate with exactly the observed occupied cells."""
    return next(
        (body for body in candidates if len(body) == len(cells) and set(body) == cells), None
    )


def snake_cells(board, side):
    """Collect all visible head and body cells for one historical snake."""
    body_char = side.lower()
    return {
        (r, c)
        for r, row in enumerate(board)
        for c, ch in enumerate(row)
        if ch in (side, body_char)
    }


def snake_head(board, side, invincible=0):
    """Find the historical head using the protection-dependent symbol case."""
    head_char = side.lower() if invincible else side
    for r, row in enumerate(board):
        for c, ch in enumerate(row):
            if ch == head_char:
                return (r, c)
    return None


def body_degree(cell, cells, rows, cols):
    """Count adjacent occupied cells when ordering a complete flat snake."""
    return sum(1 for nb in neighbors(cell, rows, cols) if nb in cells)


def neighbors(pos, rows, cols, levels=None):
    """Yield bounded neighboring cells in the coordinate's dimensionality."""
    return geometry.neighbors(pos, rows, cols, levels)


def step(pos, direction):
    """Advance one coordinate along a supported direction."""
    return geometry.step(pos, direction)


def direction_between(a, b):
    """Return the action joining two adjacent cells, or None."""
    delta = tuple(end - start for start, end in zip(a, b))
    return next(
        (name for name, candidate in directions_for(a).items() if candidate == delta), None
    )


def manhattan(a, b):
    """Measure orthogonal distance across every axis."""
    return geometry.manhattan(a, b)


def in_bounds(state, pos):
    """Check all dimensions without projecting away the level."""
    return geometry.bounded(pos, geometry.dimension_sizes(state))


def strategic_blocked(state, side=None):
    """Cells that are unusable without crashing or paying a score penalty."""
    return state.occupied() | (
        set() if state.invincibility(side or state.side) else set(state.wrong_food())
    )


def legal_moves(state, side=None):
    """Return playable moves while retaining costly wall impacts as legacy actions."""
    side = side or state.side
    snake = state.body(side)
    if not snake:
        return []
    if state.invincibility(side):
        return list(directions_for(state))
    return ordinary_legal_moves(state, side, snake[0])


def ordinary_legal_moves(state, side, head):
    """Filter each geometric direction through normal collision constraints."""
    return [
        direction
        for direction in directions_for(state)
        if playable_target(state, side, step(head, direction))
    ]


def playable_target(state, side, target):
    """Reject board edges and neck reversal before resolving occupancy."""
    if not in_bounds(state, target) or target in state.body(side)[1:2]:
        return False
    if target in state.walls:
        return True
    return not target_occupied(state, side, target)


def target_occupied(state, side, target):
    """Release only a confirmed tail on a nongrowing move."""
    blocked = state.occupied()
    tail = state.tail_food_a if side == 'A' else state.tail_food_b
    eats = any(target in cells for cells in (state.food, state.shot_food, state.stars, tail))
    if not eats and side in state.reliable_tails:
        blocked.discard(state.body(side)[-1])
    return target in blocked


def available_moves(state, side=None):
    """Prefer existing non-body collision actions; retain forced v7 crashes."""
    side = side or state.side
    if state.is_terminal() or not state.body(side):
        return []
    return legal_moves(state, side) or forced_crash_directions(state.body(side))


def forced_crash_directions(body):
    """Keep possible actions when every ordinary move is blocked."""
    return [
        direction
        for direction in directions_for(body[0])
        if len(body) < 2 or step(body[0], direction) != body[1]
    ]


def available_actions(state, side=None):
    """Enumerate movement and affordable shooting actions for the actor."""
    side = side or state.side
    actions = [('move', direction) for direction in available_moves(state, side)]
    if can_attempt_shoot(state, side):
        actions.extend(('shoot', direction) for direction in directions_for(state))
    return actions


def can_attempt_shoot(state, side):
    """Require a live actor with enough points before generating shots."""
    return (
        not state.is_terminal()
        and bool(state.body(side))
        and state.scores.get(side, 0) >= SHOOT_COST
    )


def apply_action(state, action_type, direction, side=None):
    """Dispatch one action through the shared movement or shooting rules."""
    if action_type == 'move':
        return apply_move(state, direction, side)
    if action_type == 'shoot':
        return apply_shoot(state, direction, side)
    return apply_invalid_action(state, action_type, direction, side)


def apply_invalid_action(state, action_type='invalid', direction=None, side=None):
    """Invalid actions/timeouts spend a turn without ever becoming a v7 crash."""
    if state.is_terminal():
        raise ValueError('Cannot act after the game has ended')
    side = side or state.side
    scores = dict(state.scores)
    before = scores.get(side, 0)
    scores[side] = before + (0 if state.invincibility(side) else INVALID_ACTION_PENALTY)
    metadata = action_metadata(action_type, direction, before, scores[side])
    metadata['valid'] = False
    after = stationary_turn(state, side, scores=scores)
    after = count_timeout(after, state, side, action_type)
    return finish_v10_turn(state, after, side, metadata)


def count_timeout(after, before, side, action_type):
    """Count timeouts separately from action penalties and crash history."""
    if action_type != 'timeout':
        return after
    name = 'timeouts_a' if side == 'A' else 'timeouts_b'
    return replace(after, **{name: getattr(before, name) + 1})


def stationary_turn(state, side, **changes):
    """Advance the global clock and round erosion without moving the actor."""
    return replace(
        state,
        side='B' if side == 'A' else 'A',
        enemy=side,
        remaining_moves=max(0, state.remaining_moves - 1),
        reliable_tails=state.reliable_tails | {side},
        walls=walls_after_turn(state, side),
        **changes,
    )


def apply_shoot(state, direction, side=None):
    """Spend a stationary turn; enemy cuts never count as crashes."""
    state = expire_bombs(state)
    require_nonterminal(state, 'shoot')
    side = side or state.side
    return resolve_shooting_turn(state, direction, side)


def resolve_shooting_turn(state, direction, side):
    """Resolve a validated actor's stationary shot and end-turn effects."""
    if not state.body(side):
        return state
    if direction not in directions_for(state):
        return apply_invalid_action(state, 'shoot', direction, side)
    rival_side = 'B' if side == 'A' else 'A'
    before = state.scores.get(side, 0)
    valid = before >= SHOOT_COST
    scores = dict(state.scores)
    scores[side] = score_after_shoot(state, side, valid)
    metadata = action_metadata('shoot', direction, before, scores[side])
    metadata.update(shot_cost_metadata(valid))
    snakes = dict(state.snakes)
    generated = frozenset()
    if valid:
        snakes, generated = resolve_shot_hit(state, side, direction, metadata)
    after = replace(
        state,
        snakes=snakes,
        scores=scores,
        side=rival_side,
        enemy=side,
        shot_food=state.shot_food | generated,
        remaining_moves=max(0, state.remaining_moves - 1),
        reliable_tails=state.reliable_tails | {side},
        walls=walls_after_turn(state, side),
        last_action=metadata,
    )
    return finish_v10_turn(state, after, side, metadata)


def shot_cost_metadata(valid):
    """Report whether the shot fired and how many points were spent."""
    return dict(valid=valid, shoot_cost=SHOOT_COST if valid else 0)


def require_nonterminal(state, action):
    """Reject transitions after any authoritative terminal condition."""
    if state.is_terminal():
        raise ValueError(f'Cannot {action} after the game has ended')


def score_after_shoot(state, side, valid):
    """Spending applies through invincibility; invalid shots use its immunity."""
    before = state.scores.get(side, 0)
    if valid:
        return before - SHOOT_COST
    return before + (0 if state.invincibility(side) else INVALID_ACTION_PENALTY)


def resolve_shot_hit(state, side, direction, metadata):
    """Find the first observed or true rival cell on the exact straight ray."""
    rival_side = 'B' if side == 'A' else 'A'
    rival = state.observed_snake_cells.get(rival_side, state.body(rival_side))
    hit = first_shot_hit(
        state.head(side),
        directions_for(state)[direction],
        state.rows,
        state.cols,
        state.walls,
        rival,
        state.levels,
    )
    if hit is None:
        return dict(state.snakes), frozenset()
    metadata.update(hit=True, hit_coordinate=hit)
    return resolve_shot_cut(state, rival_side, hit, metadata)


def resolve_shot_cut(state, rival_side, hit, metadata):
    """Preserve immunity and mark unknown cut extent without inventing hidden cells."""
    if state.invincibility(rival_side):
        metadata['shot_blocked_by_invincibility'] = True
        return dict(state.snakes), frozenset()
    snakes, generated, cut_length = cut_rival(state, rival_side, hit)
    metadata.update(
        cut_length=cut_length,
        shot_food_generated=sorted(generated),
        cut_extent_uncertain=state.partial,
    )
    return snakes, generated


def cut_rival(state, rival_side, hit):
    """One common suffix-cut operation for shooting and invincible ramming."""
    rival = state.body(rival_side)
    if hit not in rival:
        return dict(state.snakes), frozenset(), None
    cut_index = max(1, rival.index(hit))
    detached = rival[cut_index:]
    snakes = dict(state.snakes)
    snakes[rival_side] = rival[:cut_index]
    generated = frozenset(
        cell
        for cell in detached
        if not within_blast_zone(cell, state.bombs_large, state.bombs_small)
    )
    return snakes, generated, len(detached)


def apply_move(state, direction, side=None):
    """Resolve v7 movement/content, then v8 detonation and end-turn expiry."""
    state = expire_bombs(state)
    side = side or state.side
    if not state.body(side):
        if state.is_terminal():
            raise ValueError('Cannot move after the game has ended')
        return state
    if direction not in directions_for(state):
        return apply_invalid_action(state, 'move', direction, side)
    return resolve_movement_turn(state, direction, side)


def resolve_movement_turn(state, direction, side):
    """Resolve deterministic movement before bombs and turn-dependent star effects."""
    before = state.scores.get(side, 0)
    after = _apply_move(state, direction, side)
    metadata = action_metadata('move', direction, before, after.scores.get(side, 0))
    if after.last_action is not state.last_action:
        metadata.update(after.last_action)
    moved = after.head(side) != state.head(side)
    if moved:
        after = detonate_bombs(state, after, side, metadata)
    metadata['star_consumed'] = moved and after.head(side) in state.stars
    return finish_v10_turn(state, after, side, metadata)


def detonate_bombs(before, after, side, metadata):
    """Apply one explosion penalty regardless of the number of triggered bombs."""
    head = after.head(side)
    large = triggered_bombs(head, before.bombs_large, 2)
    small = triggered_bombs(head, before.bombs_small, 1)
    if not large and not small:
        return after
    return resolve_explosion(before, after, side, large, small, metadata)


def triggered_bombs(head, bombs, radius):
    """Select actual known/true bombs within the complete coordinate distance."""
    return frozenset(bomb for bomb in bombs if geometry.chebyshev(head, bomb) <= radius)


def resolve_explosion(before, after, side, large, small, metadata):
    """Remove exploded bombs and queue matching replacements with exact scoring."""
    scores = dict(after.scores)
    score = before.scores.get(side, 0)
    scores[side] = score if before.invincibility(side) else score_after_bomb(score)
    kinds = ('large',) * len(large) + ('small',) * len(small)
    metadata.update(
        bomb_triggered=True,
        bomb_types=sorted(set(kinds)),
        bomb_type='+'.join(sorted(set(kinds))),
        bomb_positions=sorted(large | small),
        score_after=scores[side],
    )
    return replace(
        after,
        scores=scores,
        bombs_large=before.bombs_large - large,
        bombs_small=before.bombs_small - small,
        pending_bomb_respawns=before.pending_bomb_respawns + kinds,
    )


def finish_v10_turn(before, after, side, metadata):
    """Protection applies to the current action; only its actor spends a turn."""
    counter = (
        STAR_TURNS if metadata.get('star_consumed') else max(0, before.invincibility(side) - 1)
    )
    name = 'invincible_a' if side == 'A' else 'invincible_b'
    due, pending = star_clock_after_turn(before, after, side, counter)
    metadata.update(
        invincibility_before=before.invincibility(side),
        invincibility_after=counter,
        frenzy=after.star_frenzy,
    )
    return expire_bombs(
        replace(
            after,
            star_spawn_due_round=due,
            pending_star_spawns=pending,
            last_action=metadata,
            observed_snake_cells=updated_observed_bodies(before, after),
            **{name: counter},
        )
    )


def star_clock_after_turn(before, after, side, counter):
    """Select frenzy, protection or ordinary star scheduling."""
    if after.star_frenzy:
        desired = 30 if after.levels > 1 else STAR_FRENZY_COUNT
        return None, max(0, desired - len(after.stars))
    protected = {'A': after.invincible_a, 'B': after.invincible_b}
    protected[side] = counter
    if any(protected.values()):
        return None, 0
    return ordinary_star_clock(before, after)


def ordinary_star_clock(before, after):
    """Keep half-round phase when protection expires after either player's turn."""
    due = before.star_spawn_due_round
    if before.invincible_a or before.invincible_b:
        due = (MAX_MOVES - after.remaining_moves) / 2 + STAR_RESPAWN_ROUNDS
    return due, int(normal_star_ready(after, due))


def normal_star_ready(after, due):
    """A normal star waits until its deadline and an empty star slot."""
    return due is not None and (MAX_MOVES - after.remaining_moves) / 2 >= due and not after.stars


def updated_observed_bodies(before, after):
    """Update predicted confirmed prefixes while preserving unordered visible fragments."""
    return {
        side: (set(cells) - set(before.body(side))) | set(after.body(side))
        for side, cells in before.observed_snake_cells.items()
    }


def star_spawn_candidates(state):
    """Do not invent a normal star during protection, or a chosen RNG cell."""
    if not state.star_frenzy and (state.invincible_a or state.invincible_b or state.stars):
        return ()
    edible = (
        set(state.food)
        | set(state.food_values)
        | set(state.pickups)
        | set(state.shot_food)
        | set(state.tail_food_a)
        | set(state.tail_food_b)
        | set(state.stars)
    )
    return eligible_star_cells(
        state.rows,
        state.cols,
        (state.head('A'), state.head('B')),
        state.occupied(),
        edible,
        state.bombs_large,
        state.bombs_small,
        state.star_frenzy,
        levels=state.levels,
    )


def expire_bombs(state):
    """Remove bombs and pending replacements permanently at the final phase."""
    if state.remaining_moves <= BOMB_EXPIRY_MOVES and (
        state.bombs_large or state.bombs_small or state.pending_bomb_respawns
    ):
        return replace(
            state, bombs_large=frozenset(), bombs_small=frozenset(), pending_bomb_respawns=()
        )
    return state


def action_metadata(action_type, direction, before, after):
    """Local prediction only; the next server snapshot is authoritative."""
    return dict(
        action_type=action_type,
        direction=direction,
        shoot_cost=0,
        hit=False,
        hit_coordinate=None,
        cut_length=0,
        shot_food_generated=[],
        bomb_triggered=False,
        bomb_type=None,
        bomb_types=[],
        score_before=before,
        score_after=after,
        prediction=True,
        star_consumed=False,
        invincibility_before=0,
        invincibility_after=0,
        ram_cut=False,
        ram_hit_coordinate=None,
        shot_blocked_by_invincibility=False,
        frenzy=False,
    )


def _apply_move(state, direction, side=None):
    """Resolve one geometric move before explosions and protection timers."""
    if state.is_terminal():
        raise ValueError('Cannot move after the game has ended')
    side = side or state.side
    snake = state.body(side)
    if not snake:
        return state
    target = step(snake[0], direction)
    return move_to_cell(state, side, direction, target)


def move_to_cell(state, side, direction, target):
    """Apply protected impacts, ordinary crashes or content consumption."""
    snake = state.body(side)
    tail = state.tail_food_a if side == 'A' else state.tail_food_b
    grows = movement_growth(state, target, tail)
    if protected_collision(state, side, target):
        return resolve_protected_impact(state, side, direction, target)
    if movement_collision(state, snake, target, grows):
        return resolve_crash(state, side, snake)
    return resolve_step(state, side, target, grows, tail)


def protected_collision(state, side, target):
    """Identify impacts that invincibility converts into stationary actions."""
    return state.invincibility(side) and (
        not in_bounds(state, target) or target in state.occupied()
    )


def movement_growth(state, target, tail):
    """Determine whether entering a cell extends the snake this turn."""
    return any(
        target in cells
        for cells in (state.food, state.food_values, tail, state.shot_food, state.stars)
    )


def movement_collision(state, snake, target, grows):
    """Check edges, neck reversal and occupied cells with tail release."""
    occupied = state.occupied()
    if not grows and not state.partial:
        occupied.discard(snake[-1])
    # Direct calls can attempt a neck reversal even when the bot filters it.
    return not in_bounds(state, target) or target in occupied or target in snake[1:2]


def resolve_protected_impact(state, side, direction, target):
    """Stay stationary on impact and cut a rival when ramming it."""
    rival = 'B' if side == 'A' else 'A'
    score = state.scores.get(side, 0)
    metadata = action_metadata('move', direction, score, score)
    if target not in state.body(rival):
        return stationary_turn(state, side, last_action=metadata)
    snakes, generated, cut_length = cut_rival(state, rival, target)
    metadata.update(
        ram_cut=True,
        ram_hit_coordinate=target,
        cut_length=cut_length,
        shot_food_generated=sorted(generated),
    )
    return stationary_turn(
        state,
        side,
        snakes=snakes,
        shot_food=state.shot_food | generated,
        last_action=metadata,
    )


def resolve_crash(state, side, snake):
    """Keep the head prefix, update crash history and generate eligible tail food."""
    scores = dict(state.scores)
    prior = state.crashed_a if side == 'A' else state.crashed_b
    scores[side] = score_after_crash(scores.get(side, 0), prior)
    snakes = dict(state.snakes)
    snakes[side] = snake[:CRASH_KEEP_LENGTH]
    detached = detached_tail_food(state, snake)
    return stationary_turn(
        state,
        side,
        snakes=snakes,
        scores=scores,
        **crash_changes(state, side, detached),
    )


def detached_tail_food(state, snake):
    """Discard detached cells inside known or true bomb blast zones."""
    return frozenset(
        cell
        for cell in snake[CRASH_KEEP_LENGTH:]
        if not within_blast_zone(cell, state.bombs_large, state.bombs_small)
    )


def crash_changes(state, side, detached):
    """Assign detached tail food to the opponent and record the crashing actor."""
    if side == 'A':
        return {'crashed_a': True, 'tail_food_b': state.tail_food_b | detached}
    return {'crashed_b': True, 'tail_food_a': state.tail_food_a | detached}


def resolve_step(state, side, target, grows, tail):
    """Advance the body and apply the destination cell effects."""
    wall = target in state.walls
    eats_shot = target in state.shot_food
    eats = target in state.food and not eats_shot
    move_score, multipliers = resolve_step_score(state, side, target, tail, eats)
    scores = dict(state.scores)
    scores[side] = scores.get(side, 0) + move_score
    snakes = dict(state.snakes)
    snakes[side] = next_snake_body(state.body(side), target, wall, grows)
    food_updates = update_numbered_food(state, target, eats, wall)
    pickups = update_pickups(state.pickups, target, wall)
    return finish_step(state, side, target, snakes, scores, multipliers, food_updates, pickups)


def next_snake_body(snake, target, wall, grows):
    """Build the moved or grown body while preserving its ordered prefix."""
    if wall:
        return snake
    return (target,) + (snake if grows else snake[:-1])


def resolve_step_score(state, side, target, tail, eats):
    """Apply food, multiplier, wrong-digit and ordinary movement scoring."""
    multipliers = dict(state.multipliers)
    multiplier = multipliers.get(side, 1)
    if target in state.walls:
        return WALL_HIT_PENALTY, multipliers
    if target in tail:
        return tail_food_reward(multiplier), multipliers
    if target in state.shot_food:
        return shot_food_reward(multiplier), multipliers
    if target in state.stars:
        return FOOD_SCORE * multiplier, multipliers
    return score_board_objective(state, side, target, eats, multipliers)


def score_board_objective(state, side, target, eats, multipliers):
    """Score a consumed objective using the pre-consumption multiplier."""
    if eats:
        return state.food_reward(side, target), multipliers
    if target in state.food_values:
        return wrong_digit_score(state, side), multipliers
    if target in state.pickups:
        multipliers[side] = max(1, multipliers.get(side, 1)) + 1
        return MULTIPLIER_PICKUP_SCORE, multipliers
    return NORMAL_SCORE, multipliers


def wrong_digit_score(state, side):
    """Apply the wrong-digit penalty unless the actor is invincible."""
    return 0 if state.invincibility(side) else WRONG_FOOD_PENALTY


def update_pickups(pickups, target, wall):
    """Remove only the pickup actually consumed by a moving actor."""
    if target in pickups and not wall:
        return frozenset(pickups - {target})
    return pickups


def update_numbered_food(state, target, eats, wall):
    """Consume copies and advance the cyclic food sequence correctly."""
    values = dict(state.food_values)
    copies = dict(state.food_copy_targets)
    eaten_digit = values.get(target)
    consumes = eats and not wall
    cleared_digit = eaten_digit if consumes else None
    if not wall:
        values = consume_numbered_food(values, copies, target, cleared_digit)
    next_digit = next_digit_after_move(state.next_food_digit, values, consumes)
    return {
        'food': current_food_positions(state, values, next_digit, target, cleared_digit),
        'food_values': values,
        'food_copy_targets': copies,
        'next_food_digit': next_digit,
    }


def consume_numbered_food(values, copies, target, digit):
    """Remove all target copies or one wrong copy with its replacement count."""
    if digit is None:
        # A wrong digit consumes one copy. Its replacement is server-owned.
        values.pop(target, None)
        return values
    copies.pop(digit, None)
    return {position: value for position, value in values.items() if value != digit}


def next_digit_after_move(digit, values, consumes):
    """Advance the target only when the current ordered digit was consumed."""
    if consumes and digit is not None:
        return cyclic_digit(digit)
    if values and digit is None:
        return find_next_food_digit(values.values())
    return digit


def current_food_positions(state, values, next_digit, target, cleared_digit):
    """Recompute playable target positions after a numbered-food transition."""
    if values:
        return positions_with_digit(values, next_digit)
    return frozenset(
        cell
        for cell in state.food
        if cell != target and not is_cleared_digit(state, cell, cleared_digit)
    )


def is_cleared_digit(state, cell, digit):
    """Identify a remaining copy removed by consuming the active target."""
    return digit is not None and state.food_values.get(cell) == digit


def finish_step(state, side, target, snakes, scores, multipliers, food, pickups):
    """Construct the next state while retaining geometry and observation metadata."""
    return replace(
        state,
        rows=state.rows,
        cols=state.cols,
        board=state.board,
        side=state.enemy if side == state.side else state.side,
        enemy=side,
        snakes=snakes,
        scores=scores,
        remaining_moves=max(0, state.remaining_moves - 1),
        reliable_tails=state.reliable_tails | {side},
        pickups=pickups,
        multipliers=multipliers,
        walls=walls_after_turn(state, side),
        tail_food_a=state.tail_food_a - {target},
        tail_food_b=state.tail_food_b - {target},
        crashed_a=state.crashed_a,
        crashed_b=state.crashed_b,
        bombs_large=state.bombs_large,
        bombs_small=state.bombs_small,
        shot_food=state.shot_food - {target},
        pending_bomb_respawns=state.pending_bomb_respawns,
        stars=state.stars - {target},
        invincible_a=state.invincible_a,
        invincible_b=state.invincible_b,
        star_spawn_due_round=state.star_spawn_due_round,
        pending_star_spawns=state.pending_star_spawns,
        timeouts_a=state.timeouts_a,
        timeouts_b=state.timeouts_b,
        **food,
    )


def shrink_wall(walls):
    """Erode a true line or plate wall by one complete outer layer."""
    walls = frozenset(walls)
    if len(walls) <= 1:
        return frozenset()
    if len(next(iter(walls))) == 3:
        return shrink_plate(walls)
    return shrink_flat_wall(walls)


def shrink_flat_wall(walls):
    """Remove both endpoints from an axis-aligned historical line wall."""
    rows = {r for r, _ in walls}
    cols = {c for _, c in walls}
    if len(rows) == 1:
        ordered = sorted(walls, key=lambda cell: cell[1])
    elif len(cols) == 1:
        ordered = sorted(walls, key=lambda cell: cell[0])
    else:
        return walls
    return frozenset(ordered[1:-1])


def shrink_plate(walls):
    """Remove the outer ring of a true axis-aligned square plate."""
    bounds = wall_bounds(walls)
    return frozenset(cell for cell in walls if plate_interior(cell, bounds))


def wall_bounds(walls):
    """Return one low/high pair for every wall coordinate axis."""
    return tuple(wall_axis_bounds(walls, axis) for axis in range(len(next(iter(walls)))))


def wall_axis_bounds(walls, axis):
    """Find the extent of a true wall along one axis."""
    values = [cell[axis] for cell in walls]
    return min(values), max(values)


def plate_interior(cell, bounds):
    """Retain fixed-axis coordinates and interior values on the plate's axes."""
    return all(low == high or low < value < high for value, (low, high) in zip(cell, bounds))


def walls_after_turn(state, side):
    """Shrink true walls only; partial slices do not establish plate orientation."""
    if side == 'B' and not state.partial:
        return shrink_wall(state.walls)
    return state.walls


def known_neighbors(state, cell):
    """Traverse observed cells only when the board has partial visibility."""
    adjacent = neighbors(cell, state.rows, state.cols, state.levels)
    if not state.partial:
        return adjacent
    return (other for other in adjacent if other in state.observation.cells)


def blocked_for_path(state, side):
    """Collect certain obstacles while releasing a confirmed head and tail."""
    blocked = state.occupied()
    if not state.invincibility(side):
        blocked.update(state.wrong_food())
    snake = state.body(side)
    if snake:
        blocked.discard(snake[0])
        if side in state.reliable_tails:
            blocked.discard(snake[-1])
    return blocked


def release_times(state):
    """Assign occupancy-release times without predicting unknown plate geometry."""
    releases = observed_body_release_times(state)
    for snake_side in ('A', 'B'):
        snake = state.body(snake_side)
        if not snake:
            continue
        add_body_release_times(releases, snake, snake_side in state.reliable_tails)
    if not state.partial:
        releases.update(wall_release_times(state.walls))
    else:
        releases.update({cell: NEVER_RELEASE for cell in state.walls})
    return releases


def observed_body_release_times(state):
    """Unordered visible fragments never acquire an invented tail-release time."""
    return {cell: NEVER_RELEASE for cells in state.observed_snake_cells.values() for cell in cells}


def add_body_release_times(releases, snake, reliable):
    """Set exact release turns only for a body whose ordering is confirmed."""
    for index, cell in enumerate(snake):
        if reliable:
            releases[cell] = min(releases.get(cell, NEVER_RELEASE), len(snake) - index)
        else:
            releases[cell] = NEVER_RELEASE


def wall_release_times(walls):
    """Derive erosion layers from true wall coordinates in either geometry."""
    if not walls:
        return {}
    if len(next(iter(walls))) == 2:
        return flat_wall_release_times(walls)
    bounds = wall_bounds(walls)
    return {cell: wall_release_layer(cell, bounds) + 2 for cell in walls}


def flat_wall_release_times(walls):
    """Preserve historical ordered-line release behavior for flat-board fixtures."""
    axis = 1 if len({cell[0] for cell in walls}) == 1 else 0
    ordered = sorted(walls, key=lambda cell: cell[axis])
    return {cell: min(index, len(ordered) - index - 1) + 2 for index, cell in enumerate(ordered)}


def wall_release_layer(cell, bounds):
    """Count complete outer rings before a cell disappears."""
    distances = [
        min(value - low, high - value) for value, (low, high) in zip(cell, bounds) if low != high
    ]
    return min(distances, default=0)


def temporal_shortest_path(state, start, goal, side, max_time=None, avoid_blast=False):
    """Find a shortest route respecting body release and protection expiry."""
    if start is None or goal is None:
        return None
    if start == goal:
        return (start,)
    traversal = TemporalTraversal(state, start, side, max_time, (goal,), avoid_blast)
    traversal.run(goal)
    return restore_path(traversal.parents, goal)


def temporal_shortest_distance(state, start, goal, side, max_time=None):
    """Return the temporal route length or None when unreachable."""
    path = temporal_shortest_path(state, start, goal, side, max_time)
    return None if path is None else len(path) - 1


def temporal_distance_map(state, start, side, max_time=None, goals=(), avoid_blast=False):
    """Compute one shared temporal traversal for multiple candidate objectives."""
    if start is None:
        return {}
    traversal = TemporalTraversal(state, start, side, max_time, goals, avoid_blast)
    traversal.run()
    return traversal.distances


class TemporalTraversal:
    """Bounded BFS over observed/true cells with explicit time-dependent blockers."""

    def __init__(self, state, start, side, max_time, goals, avoid_blast):
        """Prepare traversal geometry, blockers and per-search queue state."""
        self.state, self.side, self.start = state, side, start
        self.snake = state.body(side)
        self.goals = frozenset(goals)
        self.wrong = state.wrong_food()
        self.releases = temporal_releases(state, side, start, self.goals)
        self.max_time = max_time or state.rows * state.cols
        self.avoid_blast = avoid_blast
        self.blast_cells = traversal_blast_cells(state, avoid_blast)
        self.queue = deque([start])
        self.parents = {start: None}
        self.distances = {start: 0}
        self.goal = None

    def run(self, goal=None):
        """Stop at the requested destination, or visit the bounded reachable region."""
        self.goal = goal
        while self.queue:
            current = self.queue.popleft()
            if self.distances[current] < self.max_time and self.expand(current):
                return

    def expand(self, current):
        """Visit admissible neighbors once in the shared direction order."""
        for target in known_neighbors(self.state, current):
            if self.admissible(current, target):
                self.parents[target] = current
                self.distances[target] = self.distances[current] + 1
                if target == self.goal:
                    return True
                self.queue.append(target)
        return False

    def admissible(self, current, target):
        """Apply occupancy release, reverse-neck and hazard constraints."""
        next_time = self.distances[current] + 1
        if target in self.parents or self.releases.get(target, 0) > next_time:
            return False
        if temporal_neck_reversal(self.snake, target, next_time):
            return False
        return self.content_allowed(target, next_time)

    def content_allowed(self, target, next_time):
        """Check protection duration for wrong digits and known bomb zones."""
        if wrong_digit_blocks(self.state, self.side, target, next_time, self.goals):
            return False
        return not self.blast_blocks(target, next_time)

    def blast_blocks(self, target, next_time):
        """Use the global bomb deadline for this actor's projected move."""
        return self.avoid_blast and path_blast_blocked(
            self.state, target, self.side, next_time, self.blast_cells
        )


def temporal_releases(state, side, start, goals):
    """Build release clocks once for a temporal traversal."""
    releases = release_times(state)
    if not state.invincibility(side):
        releases.update({cell: NEVER_RELEASE for cell in state.wrong_food() if cell not in goals})
    releases[start] = 0
    return releases


def temporal_neck_reversal(snake, target, next_time):
    """The first step cannot reverse directly into the current neck."""
    return next_time == 1 and len(snake) > 1 and target == snake[1]


def wrong_digit_blocks(state, side, target, next_time, goals):
    """Unknown-order digits remain costly after invincibility expires."""
    return (
        target in state.wrong_food()
        and target not in goals
        and state.invincibility(side) < next_time
    )


def traversal_blast_cells(state, avoid_blast):
    """Materialize known local blast cubes once, only when requested."""
    if avoid_blast:
        return blast_zone_cells(
            state.rows, state.cols, state.bombs_large, state.bombs_small, state.levels
        )
    return None


def restore_path(parents, goal):
    """Reconstruct an ordered route only if its destination was reached."""
    if goal not in parents:
        return None
    path = []
    while goal is not None:
        path.append(goal)
        goal = parents[goal]
    return tuple(reversed(path))


def path_blast_blocked(state, cell, side, move_number, blast_cells=None):
    """Evaluate known bomb danger at the actors projected global move."""
    if state.invincibility(side) >= move_number:
        return False
    if not bombs_active_on_turn(state.remaining_moves, side == state.side, move_number):
        return False
    return (
        cell in blast_cells
        if blast_cells is not None
        else within_blast_zone(cell, state.bombs_large, state.bombs_small)
    )


def shortest_path(state, start, goal, side, avoid_blast=False):
    """Find a route through currently observed free cells or a true offline board."""
    if start is None or goal is None:
        return None
    traversal = StaticTraversal(state, start, side, goal, avoid_blast)
    traversal.run(goal)
    return restore_path(traversal.parents, goal)


class StaticTraversal(TemporalTraversal):
    """BFS variant whose occupancy remains fixed throughout the planned route."""

    def __init__(self, state, start, side, goal, avoid_blast):
        """Prepare traversal geometry, blockers and per-search queue state."""
        super().__init__(
            state, start, side, state.levels * state.rows * state.cols, (goal,), avoid_blast
        )
        self.blocked = blocked_for_path(state, side)
        self.blocked.discard(start)

    def admissible(self, current, target):
        """Reject visited/blocked cells before checking timed protection and bombs."""
        if target in self.parents or target in self.blocked:
            return False
        next_time = self.distances[current] + 1
        if target in self.wrong and self.state.invincibility(self.side) < next_time:
            return False
        return not self.blast_blocks(target, next_time)


def shortest_distance(state, start, goal, side):
    """Return the static route length or None when no route is known."""
    path = shortest_path(state, start, goal, side)
    return None if path is None else len(path) - 1


def flood_region(state, start, blocked=None):
    """Find the reachable known or true region from an in-bounds start."""
    if start is None or not in_bounds(state, start):
        return set()
    blocked = set(state.occupied() if blocked is None else blocked)
    blocked.discard(start)
    return walk_region(state, start, blocked)


def walk_region(state, start, blocked):
    """Traverse free neighboring cells once using a queue."""
    queue = deque([start])
    seen = {start}
    while queue:
        current = queue.popleft()
        for cell in known_neighbors(state, current):
            if cell not in seen and cell not in blocked:
                seen.add(cell)
                queue.append(cell)
    return seen


def open_adjacent_cells(state, cell, blocked):
    """Yield known neighboring cells outside the supplied obstacle set."""
    return (other for other in known_neighbors(state, cell) if other not in blocked)


def count_exits(state, side):
    """Count available neighboring exits from a snakes current head."""
    head = state.head(side)
    if head is None:
        return 0
    # Wrong digits are legal moves but do not count as penalty-free escapes.
    blocked = strategic_blocked(state, side)
    snake = state.body(side)
    if snake and side in state.reliable_tails:
        blocked.discard(snake[-1])
    return sum(1 for _ in open_adjacent_cells(state, head, blocked))
