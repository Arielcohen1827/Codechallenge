from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field, replace
from snake_bot.game_rules import (
    CRASH_PENALTY,
    CRASH_KEEP_LENGTH,
    ELIMINATION_SCORE_THRESHOLD,
    MAX_MOVES,
    FOOD_SCORE,
    MULTIPLIER_PICKUP_SCORE,
    RIVAL_CRASH_REWARD,
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
    STAR_TURNS, STAR_RESPAWN_ROUNDS, STAR_FRENZY_COUNT, MAX_TIMEOUTS,
    star_spawn_candidates as eligible_star_cells,
)


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

    def invincibility(self, side):
        return self.invincible_a if side == 'A' else self.invincible_b

    @property
    def star_frenzy(self):
        return self.remaining_moves <= BOMB_EXPIRY_MOVES

    def terminal_winner(self):
        if self.timeouts_a >= MAX_TIMEOUTS:
            return 'B'
        if self.timeouts_b >= MAX_TIMEOUTS:
            return 'A'
        for side in ('A', 'B'):
            if self.scores.get(side, 0) < ELIMINATION_SCORE_THRESHOLD:
                return 'B' if side == 'A' else 'A'
        if self.remaining_moves <= 0:
            a, b = self.scores.get('A', 0), self.scores.get('B', 0)
            return 'A' if a > b else 'B' if b > a else None
        return None

    def is_terminal(self):
        return max(self.timeouts_a, self.timeouts_b) >= MAX_TIMEOUTS or self.remaining_moves <= 0 or any(
            score < ELIMINATION_SCORE_THRESHOLD for score in self.scores.values())

    def head(self, side):
        snake = self.snakes.get(side, ())
        return snake[0] if snake else None

    def body(self, side):
        return self.snakes.get(side, ())

    def occupied(self):
        cells = set(self.walls)
        for snake in self.snakes.values():
            cells.update(snake)
        return cells

    def wrong_food(self):
        return frozenset(set(self.food_values) - set(self.food))

    def objective_cells(self):
        return frozenset(set(self.food) | set(self.pickups) | set(self.shot_food) | set(self.stars))

    def food_reward(self, side, pos):
        digit = self.food_values.get(pos)
        return food_points(digit, self.multipliers.get(side, 1))

    def numbered_sequence(self, limit=5):
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
        ordered = []
        for offset in range(9):
            digit = cyclic_digit(self.next_food_digit, offset)
            positions = positions_by_digit.get(digit)
            if positions:
                ordered.append((digit, tuple(sorted(positions))))
                if len(ordered) >= limit:
                    break
        return tuple(ordered)


def parse_board(board_text, rows=None, cols=None):
    lines = [line.rstrip('\n') for line in board_text.splitlines() if line.strip('\n')]
    cleaned = []
    for line in lines:
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]
        cleaned.append(line)
    if rows is None:
        rows = len(cleaned)
    if cols is None:
        cols = max((len(line) for line in cleaned), default=0)
    board = tuple(line.ljust(cols)[:cols] for line in cleaned[:rows])
    if len(board) < rows:
        board += tuple(' ' * cols for _ in range(rows - len(board)))
    return board, rows, cols


def parse_state(data, previous_snakes=None, crash_history=None):
    """Parse a snapshot; crash_history is caller-owned, not inferred from scores."""
    raw_rows = data.get('rows')
    raw_cols = data.get('cols')
    if (raw_rows is None or raw_cols is None) and data.get('board_size'):
        try:
            size_rows, size_cols = str(data['board_size']).lower().split('x', 1)
            raw_rows = raw_rows if raw_rows is not None else int(size_rows.strip())
            raw_cols = raw_cols if raw_cols is not None else int(size_cols.strip())
        except (TypeError, ValueError):
            pass
    board, rows, cols = parse_board(data.get('board', ''), raw_rows, raw_cols)
    side = normalize_side(data.get('side'), board)
    enemy = 'B' if side == 'A' else 'A'
    scores = {
        'A': int(data.get('score_1', data.get('score_A', 0)) or 0),
        'B': int(data.get('score_2', data.get('score_B', 0)) or 0),
    }
    shot_food = frozenset(
        (r, c)
        for r, row in enumerate(board)
        for c, ch in enumerate(row)
        if ch == '*'
    )
    food_values = {
        (r, c): int(ch)
        for r, row in enumerate(board)
        for c, ch in enumerate(row)
        if ch in '123456789'
    }
    food_copy_targets = {
        digit: sum(1 for value in food_values.values() if value == digit)
        for digit in set(food_values.values())
    }
    next_food_digit = find_next_food_digit(food_values.values())
    numbered_target = {
        pos for pos, digit in food_values.items() if digit == next_food_digit
    }
    # v9 '*' is unsequenced shot food, even when no numeric food remains.
    food = frozenset(numbered_target)
    pickups = frozenset(
        (r, c)
        for r, row in enumerate(board)
        for c, ch in enumerate(row)
        if ch == 'X'
    )
    walls = frozenset(
        (r, c)
        for r, row in enumerate(board)
        for c, ch in enumerate(row)
        if ch == '#'
    )
    invincible = {
        'A': max(0, int(data.get('invincible_1', data.get('invincible_A', 0)) or 0)),
        'B': max(0, int(data.get('invincible_2', data.get('invincible_B', 0)) or 0)),
    }
    snakes = {
        'A': reconstruct_snake(board, 'A', invincible['A']),
        'B': reconstruct_snake(board, 'B', invincible['B']),
    }
    reliable = set()
    if previous_snakes:
        for snake_side in ('A', 'B'):
            tracked = track_snake(previous_snakes.get(snake_side, ()), board, snake_side,
                                  invincible[snake_side])
            if tracked is not None:
                snakes[snake_side] = tracked
                reliable.add(snake_side)

    remaining_moves = int(data.get('remaining_moves', MAX_MOVES) or 0)
    stars = frozenset((r, c) for r, row in enumerate(board)
                      for c, ch in enumerate(row) if ch == '\u2b50')
    return GameState(
        rows=rows,
        cols=cols,
        board=board,
        side=side,
        enemy=enemy,
        snakes=snakes,
        food=food,
        scores=scores,
        remaining_moves=remaining_moves,
        reliable_tails=frozenset(reliable),
        food_values=food_values,
        next_food_digit=next_food_digit,
        pickups=pickups,
        multipliers={
            'A': max(1, int(data.get('multiplier_1', data.get('multiplier_A', 1)) or 1)),
            'B': max(1, int(data.get('multiplier_2', data.get('multiplier_B', 1)) or 1)),
        },
        walls=walls,
        food_copy_targets=food_copy_targets,
        tail_food_a=frozenset((r, c) for r, row in enumerate(board)
                              for c, ch in enumerate(row) if ch == '\u24b6'),
        tail_food_b=frozenset((r, c) for r, row in enumerate(board)
                              for c, ch in enumerate(row) if ch == '\u24b7'),
        crashed_a=(crash_history or {}).get('A'),
        crashed_b=(crash_history or {}).get('B'),
        bombs_large=frozenset((r, c) for r, row in enumerate(board)
                               for c, ch in enumerate(row) if ch == '💣')
                    if remaining_moves > BOMB_EXPIRY_MOVES else frozenset(),
        bombs_small=frozenset((r, c) for r, row in enumerate(board)
                               for c, ch in enumerate(row) if ch == '🧨')
                    if remaining_moves > BOMB_EXPIRY_MOVES else frozenset(),
        shot_food=shot_food,
        stars=stars,
        pending_star_spawns=(max(0, STAR_FRENZY_COUNT - len(stars))
                             if remaining_moves <= BOMB_EXPIRY_MOVES else 0),
        invincible_a=invincible['A'], invincible_b=invincible['B'],
        # A zero-counter midgame snapshot cannot reveal the previous expiry
        # round. Do not fabricate the next server star's deadline.
        star_spawn_due_round=(STAR_RESPAWN_ROUNDS if remaining_moves >= MAX_MOVES -
                              2 * STAR_RESPAWN_ROUNDS and not any(invincible.values()) else None),
        timeouts_a=max(0, int(data.get('timeouts_1', data.get('timeouts_A', 0)) or 0)),
        timeouts_b=max(0, int(data.get('timeouts_2', data.get('timeouts_B', 0)) or 0)),
    )


def normalize_side(raw_side, board):
    side = str(raw_side or 'A').upper()
    if side in ('A', 'B'):
        return side
    if any('A' in row for row in board):
        return 'A'
    if any('B' in row for row in board):
        return 'B'
    return 'A'


def reconstruct_snake(board, side, invincible=0):
    head = None
    cells = set()
    head_char = side.lower() if invincible else side
    body_char = side if invincible else side.lower()
    for r, row in enumerate(board):
        for c, ch in enumerate(row):
            if ch == head_char:
                head = (r, c)
                cells.add((r, c))
            elif ch == body_char:
                cells.add((r, c))
    if head is None:
        return ()

    ordered = [head]
    unused = set(cells)
    unused.remove(head)
    previous = None
    current = head
    while unused:
        candidates = [
            nb
            for nb in neighbors(current, len(board), len(board[0]) if board else 0)
            if nb in unused and nb != previous
        ]
        if not candidates:
            break
        candidates.sort(key=lambda cell: body_degree(cell, unused, len(board), len(board[0]) if board else 0))
        nxt = candidates[0]
        ordered.append(nxt)
        unused.remove(nxt)
        previous, current = current, nxt
    ordered.extend(sorted(unused))
    return tuple(ordered)


def track_snake(previous, board, side, invincible=0):
    cells = snake_cells(board, side)
    head = snake_head(board, side, invincible)
    if head is None or not cells:
        return ()
    if not previous or not previous[0]:
        return None

    # Each SnakeBrain sees the board again only after the rival has moved.
    # Its own already-committed body is therefore often unchanged verbatim.
    if head == previous[0] and len(previous) == len(cells) and set(previous) == cells:
        return tuple(previous)

    # A shot can remove any suffix while leaving the head stationary.
    prefix = tuple(previous[:len(cells)])
    if head == previous[0] and set(prefix) == cells:
        return prefix

    kept = tuple(previous[:CRASH_KEEP_LENGTH])
    if head == previous[0] and len(kept) == len(cells) and set(kept) == cells:
        return kept

    if manhattan(head, previous[0]) != 1:
        return None

    same_length = (head,) + tuple(previous[:-1])
    if len(same_length) == len(cells) and set(same_length) == cells:
        return same_length

    grew = (head,) + tuple(previous)
    if len(grew) == len(cells) and set(grew) == cells:
        return grew

    return None


def snake_cells(board, side):
    body_char = side.lower()
    return {
        (r, c)
        for r, row in enumerate(board)
        for c, ch in enumerate(row)
        if ch in (side, body_char)
    }


def snake_head(board, side, invincible=0):
    head_char = side.lower() if invincible else side
    for r, row in enumerate(board):
        for c, ch in enumerate(row):
            if ch == head_char:
                return (r, c)
    return None


def body_degree(cell, cells, rows, cols):
    return sum(1 for nb in neighbors(cell, rows, cols) if nb in cells)


def neighbors(pos, rows, cols):
    r, c = pos
    for dr, dc in DIRS.values():
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            yield (nr, nc)


def step(pos, direction):
    dr, dc = DIRS[direction]
    return pos[0] + dr, pos[1] + dc


def direction_between(a, b):
    delta = (b[0] - a[0], b[1] - a[1])
    for direction, candidate in DIRS.items():
        if candidate == delta:
            return direction
    return None


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def in_bounds(state, pos):
    return 0 <= pos[0] < state.rows and 0 <= pos[1] < state.cols


def strategic_blocked(state, side=None):
    """Cells that are unusable without crashing or paying a score penalty."""
    return state.occupied() | (set() if state.invincibility(side or state.side) else set(state.wrong_food()))


def legal_moves(state, side=None):
    side = side or state.side
    snake = state.body(side)
    if not snake:
        return []
    if state.invincibility(side):
        # Rams and harmless stationary impacts are the same four move actions.
        return list(DIRS)
    occupied = state.occupied()
    legal = []
    for direction in DIRS:
        target = step(snake[0], direction)
        if not in_bounds(state, target):
            continue
        if len(snake) > 1 and target == snake[1]:
            continue
        if target in state.walls:
            legal.append(direction)
            continue
        eats = target in state.food or target in state.shot_food or target in state.stars or target in (
            state.tail_food_a if side == 'A' else state.tail_food_b)
        blocked = set(occupied)
        if not eats and side in state.reliable_tails:
            blocked.discard(snake[-1])
        if target in blocked:
            continue
        # A move that leaves no exit next turn is still legal. Keeping it in
        # the move tree matters because a rival can use exactly that move to
        # close our last escape before it ever has to move again.
        legal.append(direction)
    return legal


def available_moves(state, side=None):
    """Prefer existing non-body collision actions; retain forced v7 crashes."""
    side = side or state.side
    if state.is_terminal() or not state.body(side):
        return []
    moves = legal_moves(state, side)
    if moves:
        return moves
    body = state.body(side)
    return [direction for direction in DIRS
            if len(body) < 2 or step(body[0], direction) != body[1]]


def available_actions(state, side=None):
    side = side or state.side
    actions = [('move', direction) for direction in available_moves(state, side)]
    if not state.is_terminal() and state.body(side) and state.scores.get(side, 0) >= SHOOT_COST:
        actions.extend(('shoot', direction) for direction in DIRS)
    return actions


def apply_action(state, action_type, direction, side=None):
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
    if action_type == 'timeout':
        name = 'timeouts_a' if side == 'A' else 'timeouts_b'
        after = replace(after, **{name: getattr(state, name) + 1})
    return finish_v10_turn(state, after, side, metadata)


def stationary_turn(state, side, **changes):
    return replace(state, side='B' if side == 'A' else 'A', enemy=side,
                   remaining_moves=max(0, state.remaining_moves - 1),
                   reliable_tails=state.reliable_tails | {side},
                   walls=shrink_wall(state.walls) if side == 'B' else state.walls,
                   **changes)


def apply_shoot(state, direction, side=None):
    """Spend a stationary turn; enemy cuts never count as crashes."""
    state = expire_bombs(state)
    if state.is_terminal():
        raise ValueError('Cannot shoot after the game has ended')
    side = side or state.side
    if not state.body(side):
        return state
    if direction not in DIRS:
        return apply_invalid_action(state, 'shoot', direction, side)
    rival_side = 'B' if side == 'A' else 'A'
    before = state.scores.get(side, 0)
    valid = before >= SHOOT_COST
    scores = dict(state.scores)
    scores[side] = before - SHOOT_COST if valid else before + (
        0 if state.invincibility(side) else INVALID_ACTION_PENALTY)
    metadata = action_metadata('shoot', direction, before, scores[side])
    metadata.update(valid=valid, shoot_cost=SHOOT_COST if valid else 0)
    snakes = dict(state.snakes)
    generated = frozenset()
    if valid:
        rival = state.body(rival_side)
        hit = first_shot_hit(state.head(side), DIRS[direction], state.rows,
                             state.cols, state.walls, rival)
        if hit is not None:
            metadata.update(hit=True, hit_coordinate=hit)
            if state.invincibility(rival_side):
                metadata['shot_blocked_by_invincibility'] = True
            else:
                snakes, generated, cut_length = cut_rival(state, rival_side, hit)
                metadata.update(cut_length=cut_length, shot_food_generated=sorted(generated))
    after = replace(
        state, snakes=snakes, scores=scores,
        side=rival_side, enemy=side,
        shot_food=state.shot_food | generated,
        remaining_moves=max(0, state.remaining_moves - 1),
        reliable_tails=state.reliable_tails | {side},
        walls=shrink_wall(state.walls) if side == 'B' else state.walls,
        last_action=metadata)
    return finish_v10_turn(state, after, side, metadata)


def cut_rival(state, rival_side, hit):
    """One common suffix-cut operation for shooting and invincible ramming."""
    rival = state.body(rival_side)
    cut_index = max(1, rival.index(hit))
    detached = rival[cut_index:]
    snakes = dict(state.snakes)
    snakes[rival_side] = rival[:cut_index]
    generated = frozenset(cell for cell in detached if not within_blast_zone(
        cell, state.bombs_large, state.bombs_small))
    return snakes, generated, len(detached)


def apply_move(state, direction, side=None):
    """Resolve v7 movement/content, then v8 detonation and end-turn expiry."""
    state = expire_bombs(state)
    side = side or state.side
    if not state.body(side):
        if state.is_terminal():
            raise ValueError('Cannot move after the game has ended')
        return state
    if direction not in DIRS:
        return apply_invalid_action(state, 'move', direction, side)
    before = state.scores.get(side, 0)
    after = _apply_move(state, direction, side)
    metadata = action_metadata('move', direction, before, after.scores.get(side, 0))
    if after.last_action is not state.last_action:
        metadata.update(after.last_action)
    moved = after.head(side) != state.head(side)
    if moved:
        head = after.head(side)
        large = frozenset(bomb for bomb in state.bombs_large
                          if within_blast_zone(head, (bomb,), ()))
        small = frozenset(bomb for bomb in state.bombs_small
                          if within_blast_zone(head, (), (bomb,)))
        if large or small:
            scores = dict(after.scores)
            scores[side] = before if state.invincibility(side) else score_after_bomb(before)
            kinds = ('large',) * len(large) + ('small',) * len(small)
            after = replace(after, scores=scores,
                            bombs_large=state.bombs_large - large,
                            bombs_small=state.bombs_small - small,
                            pending_bomb_respawns=state.pending_bomb_respawns + kinds)
            metadata.update(bomb_triggered=True,
                            bomb_types=sorted(set(kinds)),
                            bomb_type='large+small' if large and small else
                                      'large' if large else 'small',
                            bomb_positions=sorted(large | small), score_after=scores[side])
    metadata['star_consumed'] = moved and after.head(side) in state.stars
    return finish_v10_turn(state, after, side, metadata)


def finish_v10_turn(before, after, side, metadata):
    """Protection applies to the current action; only its actor spends a turn."""
    counter = STAR_TURNS if metadata.get('star_consumed') else max(0, before.invincibility(side) - 1)
    name = 'invincible_a' if side == 'A' else 'invincible_b'
    next_a = counter if side == 'A' else after.invincible_a
    next_b = counter if side == 'B' else after.invincible_b
    due = before.star_spawn_due_round
    if after.star_frenzy:
        due = None
        pending = max(0, STAR_FRENZY_COUNT - len(after.stars))
    elif next_a or next_b:
        due, pending = None, 0
    else:
        if before.invincible_a or before.invincible_b:
            # Preserve half-round phase: expiry after A must wait the same
            # forty alternating actions as expiry after B, never just39.
            due = (MAX_MOVES - after.remaining_moves) / 2 + STAR_RESPAWN_ROUNDS
        ready = due is not None and (MAX_MOVES - after.remaining_moves) / 2 >= due
        pending = int(ready and not after.stars)
    metadata.update(invincibility_before=before.invincibility(side),
                    invincibility_after=counter, frenzy=after.star_frenzy)
    return expire_bombs(replace(after, star_spawn_due_round=due,
                               pending_star_spawns=pending, last_action=metadata,
                               **{name: counter}))


def star_spawn_candidates(state):
    """Do not invent a normal star during protection, or a chosen RNG cell."""
    if not state.star_frenzy and (state.invincible_a or state.invincible_b or state.stars):
        return ()
    edible = (set(state.food) | set(state.food_values) | set(state.pickups) |
              set(state.shot_food) | set(state.tail_food_a) | set(state.tail_food_b) | set(state.stars))
    return eligible_star_cells(state.rows, state.cols, (state.head('A'), state.head('B')),
                               state.occupied(), edible, state.bombs_large,
                               state.bombs_small, state.star_frenzy)


def expire_bombs(state):
    if state.remaining_moves <= BOMB_EXPIRY_MOVES and (
            state.bombs_large or state.bombs_small or state.pending_bomb_respawns):
        return replace(state, bombs_large=frozenset(), bombs_small=frozenset(),
                       pending_bomb_respawns=())
    return state


def action_metadata(action_type, direction, before, after):
    """Local prediction only; the next server snapshot is authoritative."""
    return dict(action_type=action_type, direction=direction, shoot_cost=0,
                hit=False, hit_coordinate=None, cut_length=0,
                shot_food_generated=[], bomb_triggered=False, bomb_type=None, bomb_types=[],
                score_before=before, score_after=after, prediction=True,
                star_consumed=False, invincibility_before=0, invincibility_after=0,
                ram_cut=False, ram_hit_coordinate=None,
                shot_blocked_by_invincibility=False, frenzy=False)


def _apply_move(state, direction, side=None):
    if state.is_terminal():
        raise ValueError('Cannot move after the game has ended')
    side = side or state.side
    snake = state.body(side)
    if not snake:
        return state
    target = step(snake[0], direction)
    protected = bool(state.invincibility(side))
    if protected and (not in_bounds(state, target) or target in state.occupied()):
        rival_side = 'B' if side == 'A' else 'A'
        metadata = action_metadata('move', direction, state.scores.get(side, 0), state.scores.get(side, 0))
        changes = {}
        if target in state.body(rival_side):
            snakes, generated, cut_length = cut_rival(state, rival_side, target)
            changes.update(snakes=snakes, shot_food=state.shot_food | generated)
            metadata.update(ram_cut=True, ram_hit_coordinate=target,
                            cut_length=cut_length, shot_food_generated=sorted(generated))
        return stationary_turn(state, side, last_action=metadata, **changes)
    tail_food = state.tail_food_a if side == 'A' else state.tail_food_b
    grows = (target in state.food or target in state.food_values or target in tail_food
             or target in state.shot_food or target in state.stars)
    blocked = state.occupied()
    if not grows:
        blocked.discard(snake[-1])
    # A neck reversal remains filtered by legal_moves, but a directly applied
    # self-collision must never create an overlapping body.
    crashes = not in_bounds(state, target) or target in blocked or (
        len(snake) > 1 and target == snake[1])
    if crashes:
        scores = dict(state.scores)
        scores[side] = score_after_crash(scores.get(side, 0),
                                         state.crashed_a if side == 'A' else state.crashed_b)
        snakes = dict(state.snakes)
        snakes[side] = snake[:CRASH_KEEP_LENGTH]
        detached = frozenset(cell for cell in snake[CRASH_KEEP_LENGTH:]
                             if not within_blast_zone(cell, state.bombs_large, state.bombs_small))
        return replace(
            state, snakes=snakes, scores=scores,
            side='B' if side == 'A' else 'A', enemy=side,
            remaining_moves=max(0, state.remaining_moves - 1),
            reliable_tails=state.reliable_tails | {side},
            walls=shrink_wall(state.walls) if side == 'B' else state.walls,
            tail_food_a=state.tail_food_a | (detached if side == 'B' else frozenset()),
            tail_food_b=state.tail_food_b | (detached if side == 'A' else frozenset()),
            crashed_a=True if side == 'A' else state.crashed_a,
            crashed_b=True if side == 'B' else state.crashed_b)
    hits_wall = target in state.walls
    eats_shot = target in state.shot_food
    eats_star = target in state.stars
    eats = target in state.food and not eats_shot
    wrong_food = target in state.food_values and not eats
    takes_pickup = target in state.pickups
    if hits_wall:
        new_snake = snake
    else:
        new_snake = (target,) + snake if grows else (target,) + snake[:-1]
    snakes = dict(state.snakes)
    snakes[side] = new_snake
    scores = dict(state.scores)
    multipliers = dict(state.multipliers)
    move_score = NORMAL_SCORE
    if hits_wall:
        move_score = WALL_HIT_PENALTY
    elif target in tail_food:
        move_score = tail_food_reward(multipliers.get(side, 1))
    elif eats_shot:
        move_score = shot_food_reward(multipliers.get(side, 1))
    elif eats_star:
        move_score = FOOD_SCORE * multipliers.get(side, 1)
    elif eats:
        move_score = state.food_reward(side, target)
    elif wrong_food:
        move_score = 0 if protected else WRONG_FOOD_PENALTY
    elif takes_pickup:
        move_score = MULTIPLIER_PICKUP_SCORE
        multipliers[side] = max(1, multipliers.get(side, 1)) + 1
    scores[side] = scores.get(side, 0) + move_score

    food_values = dict(state.food_values)
    food_copy_targets = dict(state.food_copy_targets)
    eaten_digit = food_values.get(target)
    if eats and not hits_wall and eaten_digit is not None:
        food_values = {
            pos: digit for pos, digit in food_values.items() if digit != eaten_digit
        }
        food_copy_targets.pop(eaten_digit, None)
    elif not hits_wall and target in food_values:
        # A wrong digit only consumes the touched copy. The server respawns its
        # replacement at an unknown position on the following real board.
        food_values.pop(target, None)
    next_food_digit = state.next_food_digit
    if eats and not hits_wall and next_food_digit is not None:
        next_food_digit = cyclic_digit(next_food_digit)
    if food_values and next_food_digit is None:
        next_food_digit = find_next_food_digit(food_values.values())
    food = frozenset(
        pos for pos, digit in food_values.items() if digit == next_food_digit
    ) if food_values else frozenset(
        pos for pos in state.food
        if pos != target and not (
            eats and not hits_wall and eaten_digit is not None
            and state.food_values.get(pos) == eaten_digit
        )
    )
    walls = shrink_wall(state.walls) if side == 'B' else state.walls
    return GameState(
        rows=state.rows,
        cols=state.cols,
        board=state.board,
        side=state.enemy if side == state.side else state.side,
        enemy=side,
        snakes=snakes,
        food=food,
        scores=scores,
        remaining_moves=max(0, state.remaining_moves - 1),
        reliable_tails=state.reliable_tails | {side},
        food_values=food_values,
        next_food_digit=next_food_digit,
        pickups=frozenset(state.pickups - {target}) if takes_pickup and not hits_wall else state.pickups,
        multipliers=multipliers,
        walls=walls,
        food_copy_targets=food_copy_targets,
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
        last_action={},
    )


def shrink_wall(walls):
    walls = frozenset(walls)
    if len(walls) <= 1:
        return frozenset()
    rows = {r for r, _ in walls}
    cols = {c for _, c in walls}
    if len(rows) == 1:
        ordered = sorted(walls, key=lambda cell: cell[1])
    elif len(cols) == 1:
        ordered = sorted(walls, key=lambda cell: cell[0])
    else:
        return walls
    return frozenset(ordered[1:-1])


def blocked_for_path(state, side):
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
    releases = {}
    for snake_side in ('A', 'B'):
        snake = state.body(snake_side)
        if not snake:
            continue
        reliable = snake_side in state.reliable_tails
        for index, cell in enumerate(snake):
            if reliable:
                releases[cell] = min(releases.get(cell, NEVER_RELEASE), len(snake) - index)
            else:
                releases[cell] = NEVER_RELEASE
    if state.walls:
        rows = {r for r, _ in state.walls}
        if len(rows) == 1:
            ordered_wall = sorted(state.walls, key=lambda cell: cell[1])
        else:
            ordered_wall = sorted(state.walls, key=lambda cell: cell[0])
        for index, cell in enumerate(ordered_wall):
            layer = min(index, len(ordered_wall) - index - 1)
            releases[cell] = min(releases.get(cell, NEVER_RELEASE), layer + 2)
    return releases


def temporal_shortest_path(state, start, goal, side, max_time=None, avoid_blast=False):
    if start is None or goal is None:
        return None
    if start == goal:
        return (start,)

    blast_cells = blast_zone_cells(state.rows, state.cols, state.bombs_large, state.bombs_small) if avoid_blast else None
    snake = state.body(side)
    blocked_release = release_times(state)
    wrong = state.wrong_food()
    for cell in wrong:
        if cell != goal and not state.invincibility(side):
            blocked_release[cell] = NEVER_RELEASE
    blocked_release[start] = 0
    max_time = max_time or state.rows * state.cols
    q = deque([start])
    distance = {start: 0}
    parent = {start: None}

    while q:
        current = q.popleft()
        time = distance[current]
        if time >= max_time:
            continue
        for nb in neighbors(current, state.rows, state.cols):
            next_time = time + 1
            if nb in parent:
                continue
            if nb in wrong and nb != goal and state.invincibility(side) < next_time:
                continue
            if avoid_blast and path_blast_blocked(state, nb, side, next_time, blast_cells):
                continue
            if time == 0 and len(snake) > 1 and nb == snake[1]:
                continue
            release_time = blocked_release.get(nb, 0)
            if release_time > next_time:
                continue
            parent[nb] = current
            distance[nb] = next_time
            if nb == goal:
                path = [nb]
                cursor = current
                while cursor is not None:
                    path.append(cursor)
                    cursor = parent[cursor]
                return tuple(reversed(path))
            q.append(nb)
    return None


def temporal_shortest_distance(state, start, goal, side, max_time=None):
    path = temporal_shortest_path(state, start, goal, side, max_time)
    return None if path is None else len(path) - 1


def temporal_distance_map(state, start, side, max_time=None, goals=(), avoid_blast=False):
    """Compute temporal distances once when several goals share a board."""
    if start is None:
        return {}
    blast_cells = blast_zone_cells(state.rows, state.cols, state.bombs_large, state.bombs_small) if avoid_blast else None
    snake = state.body(side)
    blocked_release = release_times(state)
    goals = frozenset(goals)
    wrong = state.wrong_food()
    for cell in wrong:
        if cell not in goals and not state.invincibility(side):
            blocked_release[cell] = NEVER_RELEASE
    blocked_release[start] = 0
    max_time = max_time or state.rows * state.cols
    q = deque([start])
    distances = {start: 0}
    while q:
        current = q.popleft()
        time = distances[current]
        if time >= max_time:
            continue
        for nb in neighbors(current, state.rows, state.cols):
            next_time = time + 1
            if nb in distances:
                continue
            if nb in wrong and nb not in goals and state.invincibility(side) < next_time:
                continue
            if avoid_blast and path_blast_blocked(state, nb, side, next_time, blast_cells):
                continue
            if time == 0 and len(snake) > 1 and nb == snake[1]:
                continue
            if blocked_release.get(nb, 0) > next_time:
                continue
            distances[nb] = next_time
            q.append(nb)
    return distances


def path_blast_blocked(state, cell, side, move_number, blast_cells=None):
    if state.invincibility(side) >= move_number:
        return False
    if not bombs_active_on_turn(state.remaining_moves, side == state.side, move_number):
        return False
    return (cell in blast_cells if blast_cells is not None else
            within_blast_zone(cell, state.bombs_large, state.bombs_small))


def shortest_path(state, start, goal, side, avoid_blast=False):
    blast_cells = blast_zone_cells(state.rows, state.cols, state.bombs_large, state.bombs_small) if avoid_blast else None
    blocked = blocked_for_path(state, side)
    q = deque([start])
    parent = {start: None}
    wrong = state.wrong_food()
    distance = {start: 0} if avoid_blast or (state.invincibility(side) and wrong) else None
    blocked.discard(start)
    while q:
        current = q.popleft()
        if current == goal:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return tuple(reversed(path))
        for nb in neighbors(current, state.rows, state.cols):
            if nb not in parent and nb not in blocked:
                if nb in wrong and distance is not None and state.invincibility(side) < distance[current] + 1:
                    continue
                if avoid_blast and path_blast_blocked(state, nb, side, distance[current] + 1, blast_cells):
                    continue
                parent[nb] = current
                if distance is not None:
                    distance[nb] = distance[current] + 1
                q.append(nb)
    return None


def shortest_distance(state, start, goal, side):
    path = shortest_path(state, start, goal, side)
    return None if path is None else len(path) - 1


def flood_region(state, start, blocked=None):
    if start is None or not in_bounds(state, start):
        return set()
    blocked = set(blocked if blocked is not None else state.occupied())
    blocked.discard(start)
    q = deque([start])
    seen = {start}
    while q:
        current = q.popleft()
        for nb in neighbors(current, state.rows, state.cols):
            if nb not in seen and nb not in blocked:
                seen.add(nb)
                q.append(nb)
    return seen


def count_exits(state, side):
    head = state.head(side)
    if head is None:
        return 0
    # Wrong digits are technically legal, but they cost 500 points and do not
    # represent a useful escape. Counting them as open cells made the planner
    # enter corridors whose only eventual exit was a forced penalty.
    blocked = strategic_blocked(state, side)
    snake = state.body(side)
    if snake and side in state.reliable_tails:
        blocked.discard(snake[-1])
    return sum(1 for nb in neighbors(head, state.rows, state.cols) if nb not in blocked)
