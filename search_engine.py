from dataclasses import dataclass
from time import perf_counter

from snake_state import (
    CRASH_PENALTY,
    DIRS,
    FOOD_SCORE,
    MULTIPLIER_PICKUP_SCORE,
    NORMAL_SCORE,
    RIVAL_CRASH_REWARD,
    WALL_HIT_PENALTY,
    WRONG_FOOD_PENALTY,
    cyclic_digit,
)


INF = 10**12


class SearchTimeout(Exception):
    pass


@dataclass(frozen=True, slots=True)
class CompactState:
    rows: int
    cols: int
    side: int
    bodies: tuple[tuple[int, ...], tuple[int, ...]]
    walls: tuple[int, ...]
    food: tuple[int, ...]
    food_values: tuple[tuple[int, int], ...]
    next_food_digit: int
    pickups: tuple[int, ...]
    scores: tuple[int, int]
    multipliers: tuple[int, int]
    remaining_moves: int
    reliable_tails: tuple[bool, bool]


@dataclass(frozen=True, slots=True)
class SearchResult:
    scores: dict[str, int]
    completed_depth: int
    nodes: int
    transposition_hits: int
    elapsed_ms: float


class IterativeSearchEngine:
    def __init__(self, rows, cols, time_budget_ms=90, max_depth=6):
        self.rows = rows
        self.cols = cols
        self.size = rows * cols
        self.full_mask = (1 << self.size) - 1
        self.left_edge = sum(1 << (r * cols) for r in range(rows))
        self.right_edge = sum(1 << (r * cols + cols - 1) for r in range(rows))
        self.time_budget_ms = max(1, int(time_budget_ms))
        self.max_depth = max(2, int(max_depth))
        self.deadline = 0.0
        self.nodes = 0
        self.transposition_hits = 0
        self.transposition = {}
        self.evaluations = {}
        self.root_side = 0
        self.target = None

    @classmethod
    def from_game_state(cls, state, time_budget_ms=90, max_depth=6):
        engine = cls(state.rows, state.cols, time_budget_ms, max_depth)
        compact = CompactState(
            rows=state.rows,
            cols=state.cols,
            side=0 if state.side == 'A' else 1,
            bodies=tuple(
                tuple(engine.index(cell) for cell in state.body(side))
                for side in ('A', 'B')
            ),
            walls=tuple(sorted(engine.index(cell) for cell in state.walls)),
            food=tuple(sorted(engine.index(cell) for cell in state.food)),
            food_values=tuple(
                sorted((engine.index(cell), value) for cell, value in state.food_values.items())
            ),
            next_food_digit=state.next_food_digit or 0,
            pickups=tuple(sorted(engine.index(cell) for cell in state.pickups)),
            scores=(state.scores.get('A', 0), state.scores.get('B', 0)),
            multipliers=(state.multipliers.get('A', 1), state.multipliers.get('B', 1)),
            remaining_moves=state.remaining_moves,
            reliable_tails=('A' in state.reliable_tails, 'B' in state.reliable_tails),
        )
        return engine, compact

    def index(self, cell):
        return cell[0] * self.cols + cell[1]

    def cell(self, index):
        return divmod(index, self.cols)

    def search(self, state, root_side, legal, target=None):
        started = perf_counter()
        self.deadline = started + self.time_budget_ms / 1000
        self.root_side = 0 if root_side == 'A' else 1
        self.target = None if target is None else self.index(target)
        ordered = [direction for direction in self.ordered_moves(state) if direction in legal]
        best_scores = {}
        completed_depth = 0

        for depth in range(2, self.max_depth + 1):
            current = {}
            try:
                for direction in ordered:
                    self.check_deadline()
                    after = self.apply_move(state, direction)
                    current[direction] = self.alphabeta(after, depth - 1, -INF, INF)
            except SearchTimeout:
                break
            best_scores = current
            completed_depth = depth
            ordered.sort(key=lambda direction: best_scores[direction], reverse=True)

        return SearchResult(
            scores=best_scores,
            completed_depth=completed_depth,
            nodes=self.nodes,
            transposition_hits=self.transposition_hits,
            elapsed_ms=round((perf_counter() - started) * 1000, 2),
        )

    def check_deadline(self):
        if perf_counter() >= self.deadline:
            raise SearchTimeout

    def alphabeta(self, state, depth, alpha, beta):
        self.nodes += 1
        if (self.nodes & 31) == 0:
            self.check_deadline()

        suicide_value = self.winning_suicide_value(state)
        if suicide_value is not None:
            return suicide_value
        if state.remaining_moves <= 0:
            return self.final_score_value(state)
        moves = self.legal_moves(state)
        if not moves:
            return self.terminal_value(state)
        if depth <= 0:
            return self.evaluate(state)

        key = (state, depth)
        cached = self.transposition.get(key)
        if cached is not None:
            self.transposition_hits += 1
            return cached

        maximizing = state.side == self.root_side
        value = -INF if maximizing else INF
        cutoff = False
        for direction in self.ordered_moves(state, moves):
            child = self.apply_move(state, direction)
            score = self.alphabeta(child, depth - 1, alpha, beta)
            if maximizing:
                value = max(value, score)
                alpha = max(alpha, value)
            else:
                value = min(value, score)
                beta = min(beta, value)
            if alpha >= beta:
                cutoff = True
                break
        if not cutoff:
            self.transposition[key] = value
        return value

    def occupied_mask(self, state):
        occupied = self.mask(state.walls)
        occupied |= self.mask(state.bodies[0])
        occupied |= self.mask(state.bodies[1])
        return occupied

    @staticmethod
    def mask(cells):
        value = 0
        for cell in cells:
            value |= 1 << cell
        return value

    def target_index(self, head, direction):
        row, col = self.cell(head)
        dr, dc = DIRS[direction]
        row += dr
        col += dc
        if row < 0 or col < 0 or row >= self.rows or col >= self.cols:
            return None
        return row * self.cols + col

    def legal_moves(self, state):
        body = state.bodies[state.side]
        if not body:
            return ()
        occupied = self.occupied_mask(state)
        walls = self.mask(state.walls)
        food = set(state.food)
        moves = []
        for direction in DIRS:
            target = self.target_index(body[0], direction)
            if target is None or (len(body) > 1 and target == body[1]):
                continue
            target_bit = 1 << target
            if target_bit & walls:
                moves.append(direction)
                continue
            blocked = occupied
            if target not in food and state.reliable_tails[state.side] and body:
                blocked &= ~(1 << body[-1])
            if not target_bit & blocked:
                moves.append(direction)
        return tuple(moves)

    def winning_suicide_value(self, state):
        side = state.side
        enemy = 1 - side
        if state.scores[side] + CRASH_PENALTY <= state.scores[enemy] + RIVAL_CRASH_REWARD:
            return None
        body = state.bodies[side]
        if not body:
            return None
        occupied = set(state.bodies[0]) | set(state.bodies[1])
        walls = set(state.walls)
        for direction in DIRS:
            target = self.target_index(body[0], direction)
            if target is None:
                break
            if target in walls:
                continue
            if target in occupied:
                if len(body) > 1 and target == body[1]:
                    break
                if target == body[-1] and state.reliable_tails[side]:
                    continue
                break
        else:
            return None
        scores = list(state.scores)
        scores[side] += CRASH_PENALTY
        scores[enemy] += RIVAL_CRASH_REWARD
        difference = scores[self.root_side] - scores[1 - self.root_side]
        return 500_000_000 + difference if difference > 0 else -500_000_000 + difference

    def apply_move(self, state, direction):
        side = state.side
        body = state.bodies[side]
        target = self.target_index(body[0], direction)
        walls = state.walls
        hits_wall = target in walls
        eats = target in state.food
        food_values = dict(state.food_values)
        wrong_food = target in food_values and not eats
        takes_pickup = target in state.pickups

        if hits_wall:
            new_body = body
        elif eats:
            new_body = (target,) + body
        else:
            new_body = (target,) + body[:-1]
        bodies = list(state.bodies)
        bodies[side] = new_body

        scores = list(state.scores)
        multipliers = list(state.multipliers)
        if hits_wall:
            move_score = WALL_HIT_PENALTY
        elif eats:
            digit = food_values.get(target, 1)
            move_score = digit * FOOD_SCORE * max(1, multipliers[side])
        elif wrong_food:
            move_score = WRONG_FOOD_PENALTY
        elif takes_pickup:
            move_score = MULTIPLIER_PICKUP_SCORE
            multipliers[side] = max(1, multipliers[side]) + 1
        else:
            move_score = NORMAL_SCORE
        scores[side] += move_score

        next_digit = state.next_food_digit
        eaten_digit = food_values.get(target)
        if eats and not hits_wall and eaten_digit is not None:
            food_values = {
                cell: digit for cell, digit in food_values.items() if digit != eaten_digit
            }
        elif not hits_wall and target in food_values:
            food_values.pop(target, None)
        if eats and not hits_wall and next_digit:
            next_digit = cyclic_digit(next_digit)
        if food_values:
            food = tuple(sorted(cell for cell, digit in food_values.items() if digit == next_digit))
        else:
            food = tuple(cell for cell in state.food if cell != target)

        if side == 1:
            walls = self.shrink_wall(walls)
        reliable = list(state.reliable_tails)
        reliable[side] = True
        return CompactState(
            rows=state.rows,
            cols=state.cols,
            side=1 - side,
            bodies=(tuple(bodies[0]), tuple(bodies[1])),
            walls=tuple(walls),
            food=food,
            food_values=tuple(sorted(food_values.items())),
            next_food_digit=next_digit,
            pickups=tuple(cell for cell in state.pickups if not (takes_pickup and cell == target)),
            scores=(scores[0], scores[1]),
            multipliers=(multipliers[0], multipliers[1]),
            remaining_moves=max(0, state.remaining_moves - 1),
            reliable_tails=(reliable[0], reliable[1]),
        )

    def shrink_wall(self, walls):
        if len(walls) <= 1:
            return ()
        rows = {cell // self.cols for cell in walls}
        cols = {cell % self.cols for cell in walls}
        if len(rows) == 1:
            ordered = sorted(walls, key=lambda cell: cell % self.cols)
        elif len(cols) == 1:
            ordered = sorted(walls, key=lambda cell: cell // self.cols)
        else:
            return walls
        return tuple(ordered[1:-1])

    def ordered_moves(self, state, moves=None):
        moves = moves or self.legal_moves(state)
        body = state.bodies[state.side]
        food_values = dict(state.food_values)
        food = set(state.food)
        pickups = set(state.pickups)
        walls = set(state.walls)
        center_row = (self.rows - 1) / 2
        center_col = (self.cols - 1) / 2

        def key(direction):
            target = self.target_index(body[0], direction)
            penalty = target in walls or (target in food_values and target not in food)
            reward = 0
            if target in food:
                reward = 100_000 + food_values.get(target, 1) * 10_000
            elif target in pickups:
                reward = 35_000
            row, col = self.cell(target)
            center = -int((abs(row - center_row) + abs(col - center_col)) * 100)
            target_progress = 0
            if self.target is not None:
                tr, tc = self.cell(self.target)
                target_progress = -abs(row - tr) - abs(col - tc)
            return (0 if penalty else 1, reward, target_progress, center)

        return sorted(moves, key=key, reverse=True)

    def terminal_value(self, state):
        scores = list(state.scores)
        dead = state.side
        enemy = 1 - dead
        scores[dead] += CRASH_PENALTY
        scores[enemy] += RIVAL_CRASH_REWARD
        difference = scores[self.root_side] - scores[1 - self.root_side]
        if difference > 0:
            return 500_000_000 + difference
        if difference < 0:
            return -500_000_000 + difference
        return 0

    def final_score_value(self, state):
        difference = state.scores[self.root_side] - state.scores[1 - self.root_side]
        if difference > 0:
            return 300_000_000 + difference
        if difference < 0:
            return -300_000_000 + difference
        return 0

    def evaluate(self, state):
        cached = self.evaluations.get(state)
        if cached is not None:
            return cached
        root = self.root_side
        enemy = 1 - root
        difference = state.scores[root] - state.scores[enemy]
        root_region = self.region_size(state, root)
        enemy_region = self.region_size(state, enemy)
        root_moves = len(self.legal_moves_for(state, root))
        enemy_moves = len(self.legal_moves_for(state, enemy))
        center_delta = self.center_value(state.bodies[root][0]) - self.center_value(state.bodies[enemy][0])
        value = difference * 22
        value += (root_region - enemy_region) * 70
        value += (root_moves - enemy_moves) * 1800
        value += center_delta * 120
        value += (state.multipliers[root] - state.multipliers[enemy]) * 1800
        if self.target is not None:
            root_distance = self.manhattan(state.bodies[root][0], self.target)
            enemy_distance = self.manhattan(state.bodies[enemy][0], self.target)
            value += (enemy_distance - root_distance) * 250
        self.evaluations[state] = value
        return value

    def legal_moves_for(self, state, side):
        if side == state.side:
            return self.legal_moves(state)
        swapped = CompactState(
            rows=state.rows,
            cols=state.cols,
            side=side,
            bodies=state.bodies,
            walls=state.walls,
            food=state.food,
            food_values=state.food_values,
            next_food_digit=state.next_food_digit,
            pickups=state.pickups,
            scores=state.scores,
            multipliers=state.multipliers,
            remaining_moves=state.remaining_moves,
            reliable_tails=state.reliable_tails,
        )
        return self.legal_moves(swapped)

    def expand(self, bits):
        left = (bits & ~self.left_edge) >> 1
        right = (bits & ~self.right_edge) << 1
        up = bits >> self.cols
        down = (bits << self.cols) & self.full_mask
        return left | right | up | down

    def region_size(self, state, side):
        body = state.bodies[side]
        if not body:
            return 0
        start = 1 << body[0]
        blocked = self.occupied_mask(state) & ~start
        seen = start
        frontier = start
        while frontier:
            frontier = self.expand(frontier) & ~blocked & ~seen
            seen |= frontier
        return seen.bit_count()

    def center_value(self, index):
        row, col = self.cell(index)
        return -int(abs(row - (self.rows - 1) / 2) + abs(col - (self.cols - 1) / 2))

    def manhattan(self, first, second):
        ar, ac = self.cell(first)
        br, bc = self.cell(second)
        return abs(ar - br) + abs(ac - bc)
