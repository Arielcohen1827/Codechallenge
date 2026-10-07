from collections import deque
from dataclasses import dataclass, field


DIRS = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1),
}

FOOD_SCORE = 100
NORMAL_SCORE = 1
WRONG_FOOD_PENALTY = -500
MULTIPLIER_PICKUP_SCORE = 50
WALL_HIT_PENALTY = -500
CRASH_PENALTY = -500
RIVAL_CRASH_REWARD = 1000
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
        return frozenset(set(self.food) | set(self.pickups))

    def food_reward(self, side, pos):
        digit = self.food_values.get(pos)
        base = FOOD_SCORE if digit is None else digit * FOOD_SCORE
        return base * max(1, self.multipliers.get(side, 1))

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


def parse_state(data, previous_snakes=None):
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
    legacy_food = frozenset(
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
    food = frozenset(numbered_target if food_values else legacy_food)
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
    snakes = {
        'A': reconstruct_snake(board, 'A'),
        'B': reconstruct_snake(board, 'B'),
    }
    reliable = set()
    if previous_snakes:
        for snake_side in ('A', 'B'):
            tracked = track_snake(previous_snakes.get(snake_side, ()), board, snake_side)
            if tracked is not None:
                snakes[snake_side] = tracked
                reliable.add(snake_side)

    return GameState(
        rows=rows,
        cols=cols,
        board=board,
        side=side,
        enemy=enemy,
        snakes=snakes,
        food=food,
        scores=scores,
        remaining_moves=int(data.get('remaining_moves', 0) or 0),
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
    )


def cyclic_digit(value, offset=1):
    return ((int(value) - 1 + offset) % 9) + 1


def find_next_food_digit(values):
    present = {int(value) for value in values}
    if not present:
        return None
    candidates = [digit for digit in present if cyclic_digit(digit, -1) not in present]
    if len(candidates) == 1:
        return candidates[0]
    return min(candidates or present)


def normalize_side(raw_side, board):
    side = str(raw_side or 'A').upper()
    if side in ('A', 'B'):
        return side
    if any('A' in row for row in board):
        return 'A'
    if any('B' in row for row in board):
        return 'B'
    return 'A'


def reconstruct_snake(board, side):
    head = None
    cells = set()
    body_char = side.lower()
    for r, row in enumerate(board):
        for c, ch in enumerate(row):
            if ch == side:
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


def track_snake(previous, board, side):
    cells = snake_cells(board, side)
    head = snake_head(board, side)
    if head is None or not cells:
        return ()
    if not previous or not previous[0]:
        return None

    # Each SnakeBrain sees the board again only after the rival has moved.
    # Its own already-committed body is therefore often unchanged verbatim.
    if head == previous[0] and len(previous) == len(cells) and set(previous) == cells:
        return tuple(previous)

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


def snake_head(board, side):
    for r, row in enumerate(board):
        for c, ch in enumerate(row):
            if ch == side:
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


def strategic_blocked(state):
    """Cells that are unusable without crashing or paying a score penalty."""
    return state.occupied() | set(state.wrong_food())


def legal_moves(state, side=None):
    side = side or state.side
    snake = state.body(side)
    if not snake:
        return []
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
        eats = target in state.food
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


def apply_move(state, direction, side=None):
    side = side or state.side
    snake = state.body(side)
    if not snake:
        return state
    target = step(snake[0], direction)
    hits_wall = target in state.walls
    eats = target in state.food
    wrong_food = target in state.food_values and not eats
    takes_pickup = target in state.pickups
    if hits_wall:
        new_snake = snake
    else:
        new_snake = (target,) + snake if eats else (target,) + snake[:-1]
    snakes = dict(state.snakes)
    snakes[side] = new_snake
    scores = dict(state.scores)
    multipliers = dict(state.multipliers)
    move_score = NORMAL_SCORE
    if hits_wall:
        move_score = WALL_HIT_PENALTY
    elif eats:
        move_score = state.food_reward(side, target)
    elif wrong_food:
        move_score = WRONG_FOOD_PENALTY
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
    ) if food_values else frozenset(state.food - {target})
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


def temporal_shortest_path(state, start, goal, side, max_time=None):
    if start is None or goal is None:
        return None
    if start == goal:
        return (start,)

    snake = state.body(side)
    blocked_release = release_times(state)
    for cell in state.wrong_food():
        if cell != goal:
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
            if time == 0 and len(snake) > 1 and nb == snake[1]:
                continue
            release_time = blocked_release.get(nb, 0)
            if release_time > next_time:
                continue
            if nb in parent:
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


def temporal_distance_map(state, start, side, max_time=None, goals=()):
    """Compute temporal distances once when several goals share a board."""
    if start is None:
        return {}
    snake = state.body(side)
    blocked_release = release_times(state)
    goals = frozenset(goals)
    for cell in state.wrong_food():
        if cell not in goals:
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
            if time == 0 and len(snake) > 1 and nb == snake[1]:
                continue
            if blocked_release.get(nb, 0) > next_time or nb in distances:
                continue
            distances[nb] = next_time
            q.append(nb)
    return distances


def shortest_path(state, start, goal, side):
    blocked = blocked_for_path(state, side)
    q = deque([start])
    parent = {start: None}
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
                parent[nb] = current
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
    blocked = strategic_blocked(state)
    snake = state.body(side)
    if snake and side in state.reliable_tails:
        blocked.discard(snake[-1])
    return sum(1 for nb in neighbors(head, state.rows, state.cols) if nb not in blocked)
