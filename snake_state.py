from collections import deque
from dataclasses import dataclass


DIRS = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1),
}

FOOD_SCORE = 100
NORMAL_SCORE = 1


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

    def head(self, side):
        snake = self.snakes.get(side, ())
        return snake[0] if snake else None

    def body(self, side):
        return self.snakes.get(side, ())

    def occupied(self):
        cells = set()
        for snake in self.snakes.values():
            cells.update(snake)
        return cells


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
    board, rows, cols = parse_board(data.get('board', ''), data.get('rows'), data.get('cols'))
    side = normalize_side(data.get('side'), board)
    enemy = 'B' if side == 'A' else 'A'
    scores = {
        'A': int(data.get('score_1', data.get('score_A', 0)) or 0),
        'B': int(data.get('score_2', data.get('score_B', 0)) or 0),
    }
    food = frozenset(
        (r, c)
        for r, row in enumerate(board)
        for c, ch in enumerate(row)
        if ch == '*'
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
    if not previous or not previous[0] or manhattan(head, previous[0]) != 1:
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
        eats = target in state.food
        blocked = set(occupied)
        if not eats and side in state.reliable_tails:
            blocked.discard(snake[-1])
        if target in blocked:
            continue
        after = apply_move(state, direction, side)
        head = after.head(side)
        if head is None:
            continue
        if len(after.body(side)) > 2 and count_exits(after, side) == 0:
            continue
        legal.append(direction)
    return legal


def apply_move(state, direction, side=None):
    side = side or state.side
    snake = state.body(side)
    if not snake:
        return state
    target = step(snake[0], direction)
    eats = target in state.food
    new_snake = (target,) + snake if eats else (target,) + snake[:-1]
    snakes = dict(state.snakes)
    snakes[side] = new_snake
    scores = dict(state.scores)
    scores[side] = scores.get(side, 0) + (FOOD_SCORE if eats else NORMAL_SCORE)
    return GameState(
        rows=state.rows,
        cols=state.cols,
        board=state.board,
        side=state.enemy if side == state.side else state.side,
        enemy=side,
        snakes=snakes,
        food=frozenset(state.food - {target}) if eats else state.food,
        scores=scores,
        remaining_moves=max(0, state.remaining_moves - 1),
        reliable_tails=state.reliable_tails | {side},
    )


def blocked_for_path(state, side):
    blocked = state.occupied()
    snake = state.body(side)
    if snake:
        blocked.discard(snake[0])
        if side in state.reliable_tails:
            blocked.discard(snake[-1])
    return blocked


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
    blocked = state.occupied()
    snake = state.body(side)
    if snake and side in state.reliable_tails:
        blocked.discard(snake[-1])
    return sum(1 for nb in neighbors(head, state.rows, state.cols) if nb not in blocked)
