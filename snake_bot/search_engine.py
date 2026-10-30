from dataclasses import dataclass, replace
from time import perf_counter
from typing import Optional
from snake_bot.game_rules import (
    food_points, ELIMINATION_SCORE_THRESHOLD, SHOOT_COST, shoot_directions,
    STAR_TURNS,
)

from snake_bot.snake_state import (
    DIRS,
    FOOD_SCORE,
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
    tail_food_a: tuple[int, ...] = ()
    tail_food_b: tuple[int, ...] = ()
    crashed_a: Optional[bool] = None
    crashed_b: Optional[bool] = None
    bombs_large: tuple[int, ...] = ()
    bombs_small: tuple[int, ...] = ()
    shot_food: tuple[int, ...] = ()
    pending_bomb_respawns: tuple[str, ...] = ()
    stars: tuple[int, ...] = ()
    invincible_a: int = 0
    invincible_b: int = 0
    star_spawn_due_round: Optional[float] = 20
    pending_star_spawns: int = 0
    timeouts_a: int = 0
    timeouts_b: int = 0
    food_copy_targets: tuple[tuple[int, int], ...] = ()

    def invincibility(self, side):
        return self.invincible_a if side in (0, 'A') else self.invincible_b

    @property
    def star_frenzy(self):
        return self.remaining_moves <= 100

    def terminal_winner(self):
        if self.timeouts_a >= 10:
            return 'B'
        if self.timeouts_b >= 10:
            return 'A'
        for side, score in enumerate(self.scores):
            if score < ELIMINATION_SCORE_THRESHOLD:
                return 'B' if side == 0 else 'A'
        if self.remaining_moves <= 0:
            a, b = self.scores
            return 'A' if a > b else 'B' if b > a else None
        return None

    def is_terminal(self):
        return self.timeouts_a >= 10 or self.timeouts_b >= 10 or self.remaining_moves <= 0 or any(
            score < ELIMINATION_SCORE_THRESHOLD for score in self.scores)


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
        return engine, engine.to_compact(state)

    def to_compact(self, state):
        engine = self
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
            tail_food_a=tuple(sorted(engine.index(cell) for cell in state.tail_food_a)),
            tail_food_b=tuple(sorted(engine.index(cell) for cell in state.tail_food_b)),
            crashed_a=state.crashed_a,
            crashed_b=state.crashed_b,
            bombs_large=tuple(sorted(engine.index(cell) for cell in state.bombs_large)),
            bombs_small=tuple(sorted(engine.index(cell) for cell in state.bombs_small)),
            shot_food=tuple(sorted(engine.index(cell) for cell in state.shot_food)),
            pending_bomb_respawns=state.pending_bomb_respawns,
            stars=tuple(sorted(engine.index(cell) for cell in state.stars)),
            invincible_a=state.invincible_a, invincible_b=state.invincible_b,
            star_spawn_due_round=state.star_spawn_due_round,
            pending_star_spawns=state.pending_star_spawns,
            timeouts_a=state.timeouts_a, timeouts_b=state.timeouts_b,
            food_copy_targets=tuple(sorted(state.food_copy_targets.items())),
        )
        return compact

    def index(self, cell):
        return cell[0] * self.cols + cell[1]

    def cell(self, index):
        return divmod(index, self.cols)

    def search(self, state, root_side, legal, target=None):
        started = perf_counter()
        self.deadline = started + self.time_budget_ms / 1000
        self.root_side = 0 if root_side == 'A' else 1
        self.target = None if target is None else self.index(target)
        # Existing callers pass directions; the action boundary also accepts
        # authoritative tuples and protocol dictionaries without losing moves.
        allowed = set()
        for action in legal:
            if isinstance(action, dict):
                action = (action.get('action_type', action.get('action', 'move')), action.get('direction', action.get('data', {}).get('direction')))
            if isinstance(action, tuple) and action[0] == 'move':
                action = action[1]
            allowed.add(action)
        allowed.update(action for action in self.shoot_actions(state))
        ordered = [direction for direction in self.ordered_moves(state) if direction in allowed]
        best_scores = {}
        completed_depth = 0

        for depth in range(2, self.max_depth + 1):
            current = {}
            try:
                for direction in ordered:
                    self.check_deadline()
                    current[direction] = min(
                        self.alphabeta(after, depth - 1, -INF, INF)
                        for after in self.move_outcomes(state, direction))
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

        if state.is_terminal():
            return self.terminal_value(state)
        moves = self.available_actions(state)
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
            # Unknown first-crash history is an observation uncertainty, not
            # a selectable move. Use the worst official outcome for the root.
            score = min(self.alphabeta(child, depth - 1, alpha, beta)
                        for child in self.move_outcomes(state, direction))
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
        if state.invincibility(state.side):
            # A blocked move is a stationary protected action; a rival contact
            # is a ram. Both remain one of the four ordinary directions.
            return tuple(DIRS)
        occupied = self.occupied_mask(state)
        walls = self.mask(state.walls)
        food = set(state.food) | set(state.shot_food) | set(state.stars) | set(state.tail_food_a if state.side == 0 else state.tail_food_b)
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
        """Compatibility: a score lead never makes an ordinary crash final."""
        return None

    def available_moves(self, state):
        if state.is_terminal() or not state.bodies[state.side]:
            return ()
        moves = self.legal_moves(state)
        if moves:
            return moves
        body = state.bodies[state.side]
        return tuple(direction for direction in DIRS
                     if len(body) < 2 or self.target_index(body[0], direction) != body[1])

    def move_outcomes(self, state, direction):
        history = state.crashed_a if state.side == 0 else state.crashed_b
        if history is not None or state.scores[state.side] <= 0:
            return (self.apply_move(state, direction),)
        field = 'crashed_a' if state.side == 0 else 'crashed_b'
        first = self.apply_move(replace(state, **{field: False}), direction)
        if getattr(first, field) is True:
            repeated = self.apply_move(replace(state, **{field: True}), direction)
            return (first, repeated)
        return (replace(first, **{field: None}),)

    def to_game_state(self, state):
        """Use the authoritative transition engine instead of duplicating rules."""
        from snake_bot.snake_state import GameState
        cells = lambda items: frozenset(self.cell(item) for item in items)
        side = 'A' if state.side == 0 else 'B'
        return GameState(
            rows=state.rows, cols=state.cols, board=(), side=side,
            enemy='B' if side == 'A' else 'A',
            snakes={name: tuple(self.cell(item) for item in state.bodies[index])
                    for index, name in enumerate(('A', 'B'))},
            food=cells(state.food), food_values={self.cell(c): v for c, v in state.food_values},
            next_food_digit=state.next_food_digit or None, pickups=cells(state.pickups),
            walls=cells(state.walls), scores=dict(zip(('A', 'B'), state.scores)),
            multipliers=dict(zip(('A', 'B'), state.multipliers)),
            remaining_moves=state.remaining_moves,
            reliable_tails=frozenset(name for i, name in enumerate(('A', 'B')) if state.reliable_tails[i]),
            tail_food_a=cells(state.tail_food_a), tail_food_b=cells(state.tail_food_b),
            crashed_a=state.crashed_a, crashed_b=state.crashed_b,
            bombs_large=cells(state.bombs_large), bombs_small=cells(state.bombs_small),
            shot_food=cells(state.shot_food), pending_bomb_respawns=state.pending_bomb_respawns,
            stars=cells(state.stars), invincible_a=state.invincible_a,
            invincible_b=state.invincible_b, star_spawn_due_round=state.star_spawn_due_round,
            pending_star_spawns=state.pending_star_spawns,
            timeouts_a=state.timeouts_a, timeouts_b=state.timeouts_b,
            food_copy_targets=dict(state.food_copy_targets))

    def apply_action(self, state, action_type, direction):
        from snake_bot.snake_state import apply_action
        after = apply_action(self.to_game_state(state), action_type, direction)
        return self.to_compact(after)

    def apply_move(self, state, direction):
        if isinstance(direction, tuple):
            return self.apply_action(state, *direction)
        return self.apply_action(state, 'move', direction)

    def shoot_actions(self, state):
        if (state.is_terminal() or state.scores[state.side] < SHOOT_COST
                or state.invincibility(1 - state.side)):
            return ()
        body, rival = state.bodies[state.side], state.bodies[1 - state.side]
        if not body:
            return ()
        return tuple(('shoot', direction) for direction in shoot_directions(
            self.cell(body[0]), state.rows, state.cols,
            frozenset(self.cell(cell) for cell in state.walls),
            tuple(self.cell(cell) for cell in rival)))

    def available_actions(self, state):
        return tuple(self.available_moves(state)) + self.shoot_actions(state)

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
        moves = self.available_actions(state) if moves is None else moves
        body = state.bodies[state.side]
        food_values = dict(state.food_values)
        food = set(state.food)
        pickups = set(state.pickups)
        walls = set(state.walls)
        center_row = (self.rows - 1) / 2
        center_col = (self.cols - 1) / 2

        def key(direction):
            if isinstance(direction, tuple):
                return (0, 0, 0, 0)
            target = self.target_index(body[0], direction)
            if target is None:
                return (0, 0, 0, -self.size * 100)
            protected = state.invincibility(state.side) > 0
            penalty = (target in walls or (target in food_values and target not in food)) and not protected
            reward = 0
            if target in food:
                reward = 100_000 + food_values.get(target, 1) * 10_000
            elif target in pickups:
                reward = 15_000 + self.future_food_base_points(state, state.side) * 8
            elif target in state.stars:
                reward = 100_000 + 10_000 * state.multipliers[state.side]
            elif protected and target in state.bodies[1 - state.side]:
                rival = state.bodies[1 - state.side]
                reward = max(0, len(rival) - max(1, rival.index(target))) * 10_000
            row, col = self.cell(target)
            center = -int((abs(row - center_row) + abs(col - center_col)) * 100)
            target_progress = 0
            if self.target is not None:
                tr, tc = self.cell(self.target)
                target_progress = -abs(row - tr) - abs(col - tc)
            return (0 if penalty else 1, reward, target_progress, center)

        return sorted(moves, key=key, reverse=True)

    def terminal_value(self, state):
        if not state.is_terminal():
            return self.evaluate(state)
        if (state.timeouts_a >= 10 or state.timeouts_b >= 10
                or any(score < ELIMINATION_SCORE_THRESHOLD for score in state.scores)):
            winner = state.terminal_winner()
            root = 'A' if self.root_side == 0 else 'B'
            delta = state.scores[self.root_side] - state.scores[1 - self.root_side]
            return (500_000_000 if winner == root else -500_000_000) + delta
        return self.final_score_value(state)

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
        root_moves = len(self.productive_moves_for(state, root))
        enemy_moves = len(self.productive_moves_for(state, enemy))
        center_delta = self.center_value(state.bodies[root][0]) - self.center_value(state.bodies[enemy][0])
        value = difference * 22
        value += (root_region - enemy_region) * 70
        value += (root_moves - enemy_moves) * 1800
        value += center_delta * 120
        future_base = max(
            self.future_food_base_points(state, root),
            self.future_food_base_points(state, enemy),
        )
        multiplier_value = max(1800, min(40_000, future_base * 8))
        value += (state.multipliers[root] - state.multipliers[enemy]) * multiplier_value
        if root_moves == 0:
            value -= 45_000
        if enemy_moves == 0:
            value += 45_000
        if self.target is not None:
            root_distance = self.manhattan(state.bodies[root][0], self.target)
            enemy_distance = self.manhattan(state.bodies[enemy][0], self.target)
            value += (enemy_distance - root_distance) * 250
        if state.stars or state.invincible_a or state.invincible_b:
            value += (self.star_position_value(state, root)
                      - self.star_position_value(state, enemy)) * 22
        self.evaluations[state] = value
        return value

    def star_position_value(self, state, side):
        """Price visible protection/ram opportunities on the existing point scale."""
        body, rival = state.bodies[side], state.bodies[1 - side]
        if not body or not rival:
            return 0
        turns = max(0, (state.remaining_moves + (side == state.side)) // 2)
        duration = min(state.invincibility(side), turns)
        nearest = min(self.manhattan(body[0], cell) for cell in rival)
        # Protection can avoid a first-crash wipe; ramming exposes the detached
        # cells' real 100*mult reward. Distance and the actual own-turn horizon
        # bound both rather than granting a fixed star bonus.
        history = state.crashed_a if side == 0 else state.crashed_b
        loss = max(0, state.scores[side]) if history is False else 500
        cut_food = max(0, len(rival) - 1) * 100 * state.multipliers[side]
        points = (loss + cut_food) * duration // (STAR_TURNS * max(1, nearest))
        if not state.stars or turns <= 0:
            return points
        best = 0
        for star in state.stars:
            ours = self.manhattan(body[0], star)
            theirs = self.manhattan(rival[0], star)
            # Own-turn distances with the alternating side resolving ties.
            if (ours > turns or (not state.star_frenzy and
                    (ours > theirs or (ours == theirs and side != state.side)))):
                continue
            protected_after = max(0, state.invincibility(side) - ours)
            extra = max(0, min(STAR_TURNS, turns - ours) - protected_after)
            benefit = (loss + cut_food) * extra // (STAR_TURNS * max(1, nearest))
            immediate = 100 * state.multipliers[side]
            best = max(best, (immediate + benefit) // max(1, ours))
        return points + best

    def future_food_base_points(self, state, side):
        if not state.food_values or not state.next_food_digit:
            return FOOD_SCORE
        present = {digit for _cell, digit in state.food_values}
        ordered = []
        for offset in range(9):
            digit = cyclic_digit(state.next_food_digit, offset)
            if digit in present:
                ordered.append(digit)
        turns_left = max(0, (state.remaining_moves + (1 if side == state.side else 0)) // 2)
        captures = min(len(ordered), max(1, turns_left // 8))
        return sum(food_points(digit) for digit in ordered[:captures])

    def productive_moves_for(self, state, side):
        moves = self.legal_moves_for(state, side)
        body = state.bodies[side]
        if not body:
            return ()
        food = set(state.food)
        wrong = {cell for cell, _digit in state.food_values if cell not in food}
        if state.invincibility(side):
            wrong.clear()
        walls = set(state.walls)
        return tuple(
            direction for direction in moves
            if self.target_index(body[0], direction) not in walls
            and self.target_index(body[0], direction) not in wrong
            and (not state.invincibility(side) or (
                self.target_index(body[0], direction) is not None
                and self.target_index(body[0], direction) not in body))
        )

    def legal_moves_for(self, state, side):
        if side == state.side:
            return self.legal_moves(state)
        swapped = replace(state, side=side)
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
        wrong_food = (() if state.invincibility(side) else
                      tuple(cell for cell, _digit in state.food_values if cell not in state.food))
        blocked = (self.occupied_mask(state) | self.mask(wrong_food)) & ~start
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
