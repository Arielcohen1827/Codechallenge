"""Time-bounded adversarial search using reversible cube cell indices."""

from dataclasses import dataclass, field, replace
from time import perf_counter
from typing import Optional
from snake_bot.game_rules import (
    food_points,
    ELIMINATION_SCORE_THRESHOLD,
    SHOOT_COST,
    shoot_directions,
    STAR_TURNS,
)

from snake_bot.snake_state import (
    directions_for,
    FOOD_SCORE,
    cyclic_digit,
    step,
    manhattan,
)

INF = 10**12


def observation_key(state):
    """Separate equal visible positions with different remembered information."""
    observation = getattr(state, 'observation', None)
    knowledge = getattr(state, 'knowledge', None)
    cells = getattr(observation, 'cells', {})
    remembered = getattr(knowledge, 'remembered', {})
    return (
        tuple(sorted(cells.items())),
        tuple(
            (cell, record.symbol, record.last_seen_turn)
            for cell, record in sorted(remembered.items())
        ),
        (
            getattr(knowledge, 'last_enemy_head', None),
            getattr(knowledge, 'last_enemy_turn', None),
            getattr(knowledge, 'next_food_digit', None),
        ),
    )


def observed_body_key(bodies):
    """Hash all visible segments independently from reconstructed body prefixes."""
    return tuple((side, tuple(sorted(cells))) for side, cells in sorted(bodies.items()))


def normalized_search_action(action):
    """Accept existing directions and protocol-shaped action values at one boundary."""
    if isinstance(action, dict):
        action = protocol_search_action(action)
    if isinstance(action, tuple) and action[0] == 'move':
        return action[1]
    return action


def protocol_search_action(action):
    """Read nested protocol direction fields without changing the action kind."""
    return (
        action.get('action_type', action.get('action', 'move')),
        action.get('direction', action.get('data', {}).get('direction')),
    )


def reliable_tail_names(flags):
    """Translate compact reliability flags back to player identifiers."""
    return frozenset(name for name, reliable in zip(('A', 'B'), flags) if reliable)


class SearchTimeout(Exception):
    """Stop an incomplete depth when the decision time budget expires."""

    pass


@dataclass(frozen=True, slots=True)
class CompactState:
    """Hashable search projection retaining cube geometry and observation certainty."""

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
    levels: int = 1
    partial: bool = False
    visibility_key: tuple = ()
    observation: object = field(default=None, compare=False, hash=False)
    knowledge: object = field(default=None, compare=False, hash=False)
    observed_snake_cells: tuple = ()

    def invincibility(self, side):
        """Return the current protection timer for a compact player index or name."""
        return self.invincible_a if side in (0, 'A') else self.invincible_b

    @property
    def star_frenzy(self):
        """Whether the shared endgame clock has reached the frenzy phase."""
        return self.remaining_moves <= 100

    def terminal_winner(self):
        """Resolve official timeout, score-elimination and move-limit results."""
        loser = self.eliminated_side()
        if loser is not None:
            return ('B', 'A')[loser]
        if self.remaining_moves <= 0:
            return self.score_winner()
        return None

    def eliminated_side(self):
        """Identify timeout or score elimination in official precedence order."""
        for side, timeouts in enumerate((self.timeouts_a, self.timeouts_b)):
            if timeouts >= 10:
                return side
        for side, score in enumerate(self.scores):
            if score < ELIMINATION_SCORE_THRESHOLD:
                return side
        return None

    def score_winner(self):
        """Resolve the move-limit result, retaining an exact draw."""
        a, b = self.scores
        return 'A' if a > b else 'B' if b > a else None

    def is_terminal(self):
        """Whether an official elimination or the global move limit ended play."""
        return (
            self.timeouts_a >= 10
            or self.timeouts_b >= 10
            or self.remaining_moves <= 0
            or any(score < ELIMINATION_SCORE_THRESHOLD for score in self.scores)
        )


@dataclass(frozen=True, slots=True)
class SearchResult:
    """Expose completed scores, search effort and elapsed decision time."""

    scores: dict[str, int]
    completed_depth: int
    nodes: int
    transposition_hits: int
    elapsed_ms: float


class IterativeSearchEngine:
    """Bound alpha-beta work by wall time and retain exact shared rule transitions."""

    def __init__(self, rows, cols, time_budget_ms=90, max_depth=6, levels=1):
        """Allocate reversible coordinate masks and per-decision search caches."""
        self.rows = rows
        self.cols = cols
        self.levels = levels
        self.dimensions = (levels, rows, cols) if levels > 1 else (rows, cols)
        self.size = levels * rows * cols
        self.full_mask = (1 << self.size) - 1
        self.left_edge = sum(1 << (r * cols) for r in range(rows * levels))
        self.right_edge = self.left_edge << (cols - 1)
        self.row_edge = self.level_row_edge()
        self.row_end = self.row_edge << ((rows - 1) * cols)
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
        """Build matching geometry and a compact projection of the supplied state."""
        engine = cls(state.rows, state.cols, time_budget_ms, max_depth, state.levels)
        return engine, engine.to_compact(state)

    def level_row_edge(self):
        """Bits of each level's first row, preventing row wrapping across levels."""
        row = (1 << self.cols) - 1
        return sum(row << (level * self.rows * self.cols) for level in range(self.levels))

    def to_compact(self, state):
        """Encode rule state and observation certainty without merging distinct levels."""
        engine = self
        compact = CompactState(
            rows=state.rows,
            cols=state.cols,
            side=0 if state.side == 'A' else 1,
            bodies=engine.body_indices(state),
            walls=engine.indices(state.walls),
            food=engine.indices(state.food),
            food_values=engine.food_value_indices(state.food_values),
            next_food_digit=state.next_food_digit or 0,
            pickups=engine.indices(state.pickups),
            scores=(state.scores.get('A', 0), state.scores.get('B', 0)),
            multipliers=(state.multipliers.get('A', 1), state.multipliers.get('B', 1)),
            remaining_moves=state.remaining_moves,
            reliable_tails=('A' in state.reliable_tails, 'B' in state.reliable_tails),
            tail_food_a=engine.indices(state.tail_food_a),
            tail_food_b=engine.indices(state.tail_food_b),
            crashed_a=state.crashed_a,
            crashed_b=state.crashed_b,
            bombs_large=engine.indices(state.bombs_large),
            bombs_small=engine.indices(state.bombs_small),
            shot_food=engine.indices(state.shot_food),
            pending_bomb_respawns=state.pending_bomb_respawns,
            stars=engine.indices(state.stars),
            invincible_a=state.invincible_a,
            invincible_b=state.invincible_b,
            star_spawn_due_round=state.star_spawn_due_round,
            pending_star_spawns=state.pending_star_spawns,
            timeouts_a=state.timeouts_a,
            timeouts_b=state.timeouts_b,
            food_copy_targets=tuple(sorted(state.food_copy_targets.items())),
            levels=state.levels,
            partial=state.partial,
            visibility_key=observation_key(state),
            observation=state.observation,
            knowledge=state.knowledge,
            observed_snake_cells=observed_body_key(state.observed_snake_cells),
        )
        return compact

    def indices(self, cells):
        """Canonical sorted indices for an unordered coordinate collection."""
        return tuple(sorted(self.index(cell) for cell in cells))

    def body_indices(self, state):
        """Preserve each snake's head-to-tail ordering while encoding positions."""
        return tuple(tuple(map(self.index, state.body(side))) for side in ('A', 'B'))

    def food_value_indices(self, values):
        """Retain digit values alongside their distinct cube cell indices."""
        return tuple(sorted((self.index(cell), value) for cell, value in values.items()))

    def index(self, cell):
        """Encode each axis without collisions between rows or levels."""
        value = 0
        for component, size in zip(cell, self.dimensions):
            value = value * size + component
        return value

    def cell(self, index):
        """Invert a compact index using the engine's dimensional geometry."""
        values = []
        for size in reversed(self.dimensions):
            index, component = divmod(index, size)
            values.append(component)
        return tuple(reversed(values))

    def search(self, state, root_side, legal, target=None):
        """Return the last complete search depth within the configured decision budget."""
        started = perf_counter()
        self.start_search(root_side, target, started)
        legal = tuple(map(normalized_search_action, legal))
        if state.partial:
            return self.search_observed_moves(state, legal, started)
        return self.search_complete_state(state, legal, started)

    def start_search(self, root_side, target, started):
        """Initialize one decision's deadline and perspective."""
        self.deadline = started + self.time_budget_ms / 1000
        self.root_side = 0 if root_side == 'A' else 1
        self.target = None if target is None else self.index(target)

    def search_complete_state(self, state, legal, started):
        """Iteratively deepen only where full deterministic state is available."""
        # Existing callers pass directions; the action boundary also accepts
        # authoritative tuples and protocol dictionaries without losing moves.
        ordered = self.root_actions(state, legal)
        best_scores = {}
        completed_depth = 0

        for depth in range(2, self.max_depth + 1):
            current = {}
            try:
                current = self.search_root_depth(state, ordered, depth)
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

    def root_actions(self, state, legal):
        """Add only productive shots to caller-approved root moves."""
        allowed = set(legal) | set(self.shoot_actions(state))
        return [action for action in self.ordered_moves(state) if action in allowed]

    def search_root_depth(self, state, ordered, depth):
        """Finish an entire root depth before replacing the last complete result."""
        current = {}
        for direction in ordered:
            self.check_deadline()
            current[direction] = min(
                self.alphabeta(after, depth - 1, -INF, INF)
                for after in self.move_outcomes(state, direction)
            )
        return current

    def search_observed_moves(self, state, legal, started):
        """Score only our observed outcomes when hidden rival replies are unknowable."""
        scores = {}
        actions = tuple(dict.fromkeys(tuple(legal) + self.shoot_actions(state)))
        for action in actions:
            self.check_deadline()
            outcomes = self.move_outcomes(state, action)
            scores[action] = min(self.observed_value(after) for after in outcomes)
            self.nodes += len(outcomes)
        return SearchResult(scores, 1, self.nodes, 0, round((perf_counter() - started) * 1000, 2))

    def observed_value(self, state):
        """Use confirmed score, position and target progress without unseen enemy claims."""
        body = state.bodies[self.root_side]
        if not body:
            return -INF
        value = state.scores[self.root_side] * 22 + self.center_value(body[0]) * 120
        if self.target is not None:
            value -= self.manhattan(body[0], self.target) * 250
        return value

    def check_deadline(self):
        """Interrupt work when the current decision budget is exhausted."""
        if perf_counter() >= self.deadline:
            raise SearchTimeout

    def alphabeta(self, state, depth, alpha, beta):
        """Search alternating full-state replies with exact transposition caching."""
        self.nodes += 1
        if (self.nodes & 31) == 0:
            self.check_deadline()
        moves = self.available_actions(state)
        leaf = self.leaf_value(state, depth, moves)
        if leaf is not None:
            return leaf
        key = (state, depth)
        cached = self.transposition.get(key)
        if cached is not None:
            self.transposition_hits += 1
            return cached

        return self.search_children(state, moves, depth, alpha, beta, key)

    def leaf_value(self, state, depth, moves):
        """Resolve terminal and depth-limit leaves before using transpositions."""
        if state.is_terminal() or not moves:
            return self.terminal_value(state)
        if depth <= 0:
            return self.evaluate(state)
        return None

    def search_children(self, state, moves, depth, alpha, beta, key):
        """Store exact values only when no alpha-beta cutoff discarded replies."""
        maximizing = state.side == self.root_side
        value = -INF if maximizing else INF
        cutoff = False
        for direction in self.ordered_moves(state, moves):
            # Unknown first-crash history is an observation uncertainty, not
            # a selectable move. Use the worst official outcome for the root.
            score = self.action_value(state, direction, depth, alpha, beta)
            value, alpha, beta = self.updated_bounds(maximizing, value, score, alpha, beta)
            if alpha >= beta:
                cutoff = True
                break
        if not cutoff:
            self.transposition[key] = value
        return value

    def action_value(self, state, direction, depth, alpha, beta):
        """Treat unknown crash histories as conservative outcomes of one action."""
        return min(
            self.alphabeta(child, depth - 1, alpha, beta)
            for child in self.move_outcomes(state, direction)
        )

    @staticmethod
    def updated_bounds(maximizing, value, score, alpha, beta):
        """Apply the current actor's maximizing or minimizing bound update."""
        if maximizing:
            value = max(value, score)
            return value, max(alpha, value), beta
        value = min(value, score)
        return value, alpha, min(beta, value)

    def occupied_mask(self, state):
        """Combine all confirmed snake and wall cells into an occupancy bitset."""
        occupied = self.mask(state.walls)
        occupied |= self.mask(state.bodies[0])
        occupied |= self.mask(state.bodies[1])
        occupied |= self.observed_body_mask(state)
        return occupied

    def observed_body_mask(self, state):
        """Visible disconnected fragments remain occupied even without a complete body path."""
        return self.mask(
            self.index(cell) for _side, cells in state.observed_snake_cells for cell in cells
        )

    @staticmethod
    def mask(cells):
        """Encode an iterable of cell indices as a Python integer bitset."""
        value = 0
        for cell in cells:
            value |= 1 << cell
        return value

    def target_index(self, head, direction):
        """Return the adjacent index, or None for a move through a board face."""
        target = step(self.cell(head), direction)
        if not all(0 <= axis < size for axis, size in zip(target, self.dimensions)):
            return None
        return self.index(target)

    def legal_moves(self, state):
        """Return allowed directions, retaining protected stationary and ram actions."""
        body = state.bodies[state.side]
        if not body:
            return ()
        if state.invincibility(state.side):
            # A blocked move is a stationary protected action; a rival contact
            # is a ram. Both remain one of the four ordinary directions.
            return tuple(directions_for(self.cell(body[0])))
        return tuple(
            direction
            for direction in directions_for(self.cell(body[0]))
            if self.legal_target(state, self.target_index(body[0], direction))
        )

    def legal_target(self, state, target):
        """Keep intentional wall actions while rejecting bounds, reversal and body hits."""
        body = state.bodies[state.side]
        if target is None or (len(body) > 1 and target == body[1]):
            return False
        if target in state.walls:
            return True
        return not (1 << target) & self.blocked_for_target(state, target)

    def blocked_for_target(self, state, target):
        """Release a reliable own tail only when the arriving move will not grow."""
        blocked = self.occupied_mask(state)
        if state.reliable_tails[state.side] and target not in self.growth_cells(state):
            blocked &= ~(1 << state.bodies[state.side][-1])
        return blocked

    @staticmethod
    def growth_cells(state):
        """Known food that retains the current tail on the actor's next step."""
        tail = state.tail_food_a if state.side == 0 else state.tail_food_b
        return set(state.food) | set(state.shot_food) | set(state.stars) | set(tail)

    def winning_suicide_value(self, state):
        """Compatibility: a score lead never makes an ordinary crash final."""
        return None

    def available_moves(self, state):
        """Retain non-reversing crash actions when ordinary movement is blocked."""
        if state.is_terminal() or not state.bodies[state.side]:
            return ()
        moves = self.legal_moves(state)
        if moves:
            return moves
        body = state.bodies[state.side]
        return self.forced_directions(body)

    def forced_directions(self, body):
        """Offer crash actions only when productive moves have all disappeared."""
        return tuple(
            direction
            for direction in directions_for(self.cell(body[0]))
            if len(body) < 2 or self.target_index(body[0], direction) != body[1]
        )

    def move_outcomes(self, state, direction):
        """Preserve both official outcomes when first-crash history is unknown."""
        field = ('crashed_a', 'crashed_b')[state.side]
        if getattr(state, field) is not None or state.scores[state.side] <= 0:
            return (self.apply_move(state, direction),)
        first = self.apply_move(replace(state, **{field: False}), direction)
        if getattr(first, field) is True:
            repeated = self.apply_move(replace(state, **{field: True}), direction)
            return (first, repeated)
        return (replace(first, **{field: None}),)

    def to_game_state(self, state):
        """Use the authoritative transition engine instead of duplicating rules."""
        from snake_bot.snake_state import GameState

        def cells(items):
            """Decode an unordered compact cell collection into coordinate tuples."""
            return frozenset(map(self.cell, items))

        side = 'A' if state.side == 0 else 'B'
        return GameState(
            rows=state.rows,
            cols=state.cols,
            board=(),
            side=side,
            enemy='B' if side == 'A' else 'A',
            snakes=self.restore_snakes(state),
            food=cells(state.food),
            food_values={self.cell(c): v for c, v in state.food_values},
            next_food_digit=state.next_food_digit or None,
            pickups=cells(state.pickups),
            walls=cells(state.walls),
            scores=dict(zip(('A', 'B'), state.scores)),
            multipliers=dict(zip(('A', 'B'), state.multipliers)),
            remaining_moves=state.remaining_moves,
            reliable_tails=reliable_tail_names(state.reliable_tails),
            tail_food_a=cells(state.tail_food_a),
            tail_food_b=cells(state.tail_food_b),
            crashed_a=state.crashed_a,
            crashed_b=state.crashed_b,
            bombs_large=cells(state.bombs_large),
            bombs_small=cells(state.bombs_small),
            shot_food=cells(state.shot_food),
            pending_bomb_respawns=state.pending_bomb_respawns,
            stars=cells(state.stars),
            invincible_a=state.invincible_a,
            invincible_b=state.invincible_b,
            star_spawn_due_round=state.star_spawn_due_round,
            pending_star_spawns=state.pending_star_spawns,
            timeouts_a=state.timeouts_a,
            timeouts_b=state.timeouts_b,
            food_copy_targets=dict(state.food_copy_targets),
            levels=state.levels,
            partial=state.partial,
            observation=state.observation,
            knowledge=state.knowledge,
            observed_snake_cells=dict(state.observed_snake_cells),
        )

    def restore_snakes(self, state):
        """Decode head-to-tail paths without losing their level coordinate."""
        return {
            name: tuple(map(self.cell, state.bodies[index]))
            for index, name in enumerate(('A', 'B'))
        }

    def apply_action(self, state, action_type, direction):
        """Project one action through the authoritative GameState rule engine."""
        from snake_bot.snake_state import apply_action

        after = apply_action(self.to_game_state(state), action_type, direction)
        return self.to_compact(after)

    def apply_move(self, state, direction):
        """Accept historical direction callers and explicit action tuples."""
        if isinstance(direction, tuple):
            return self.apply_action(state, *direction)
        return self.apply_action(state, 'move', direction)

    def shoot_actions(self, state):
        """Generate only affordable rays that currently intersect a vulnerable rival."""
        if not self.can_select_shoot(state):
            return ()
        body, rival = state.bodies[state.side], state.bodies[1 - state.side]
        if not body:
            return ()
        targets = tuple(map(self.cell, rival))
        walls = frozenset(map(self.cell, state.walls))
        return tuple(
            ('shoot', direction)
            for direction in shoot_directions(
                self.cell(body[0]), state.rows, state.cols, walls, targets
            )
        )

    @staticmethod
    def can_select_shoot(state):
        """Keep unaffordable actions and invincible targets out of branching."""
        return (
            not state.is_terminal()
            and state.scores[state.side] >= SHOOT_COST
            and not state.invincibility(1 - state.side)
        )

    def available_actions(self, state):
        """Combine ordinary movement with selected useful shooting actions."""
        return tuple(self.available_moves(state)) + self.shoot_actions(state)

    def shrink_wall(self, walls):
        """Use the shared geometry-aware wall shrink rule for lines and plates."""
        from snake_bot.snake_state import shrink_wall

        return self.indices(shrink_wall(tuple(map(self.cell, walls))))

    def ordered_moves(self, state, moves=None):
        """Order legal actions with the accepted food, target and centre priorities."""
        moves = self.available_actions(state) if moves is None else moves
        return sorted(
            moves, key=lambda direction: self.move_order_key(state, direction), reverse=True
        )

    def move_order_key(self, state, direction):
        """Compare penalty, objective reward, target progress and centre distance."""
        if isinstance(direction, tuple):
            return (0, 0, 0, 0)
        target = self.target_index(state.bodies[state.side][0], direction)
        if target is None:
            return (0, 0, 0, -self.size * 100)
        progress = 0 if self.target is None else -self.manhattan(target, self.target)
        return (
            int(not self.target_penalized(state, target)),
            self.order_reward(state, target),
            progress,
            -int(self.center_distance(target) * 100),
        )

    @staticmethod
    def target_penalized(state, target):
        """Whether ordinary collision or wrong-food penalties affect this target."""
        values = dict(state.food_values)
        return (
            target in state.walls or (target in values and target not in state.food)
        ) and not state.invincibility(state.side)

    def order_reward(self, state, target):
        """Preserve food, multiplier, star and protected-ram ordering priorities."""
        values = dict(state.food_values)
        if target in state.food:
            return 100_000 + values.get(target, 1) * 10_000
        if target in state.pickups:
            return 15_000 + self.future_food_base_points(state, state.side) * 8
        if target in state.stars:
            return 100_000 + 10_000 * state.multipliers[state.side]
        return self.ram_order_reward(state, target)

    @staticmethod
    def ram_order_reward(state, target):
        """Value only rival cells that an invincible actor can actually contact."""
        rival = state.bodies[1 - state.side]
        if not state.invincibility(state.side) or target not in rival:
            return 0
        return max(0, len(rival) - max(1, rival.index(target))) * 10_000

    def terminal_value(self, state):
        """Prioritize official wins and losses ahead of ordinary positional scores."""
        if not state.is_terminal():
            return self.evaluate(state)
        if state.eliminated_side() is not None:
            winner = state.terminal_winner()
            root = 'A' if self.root_side == 0 else 'B'
            delta = state.scores[self.root_side] - state.scores[1 - self.root_side]
            return (500_000_000 if winner == root else -500_000_000) + delta
        return self.final_score_value(state)

    def final_score_value(self, state):
        """Resolve the global move-limit score difference including exact draws."""
        difference = state.scores[self.root_side] - state.scores[1 - self.root_side]
        if difference > 0:
            return 300_000_000 + difference
        if difference < 0:
            return -300_000_000 + difference
        return 0

    def evaluate(self, state):
        """Cache the economic and spatial evaluation of a complete compact state."""
        cached = self.evaluations.get(state)
        if cached is not None:
            return cached
        value = self.evaluate_position(state)
        self.evaluations[state] = value
        return value

    def evaluate_position(self, state):
        """Combine the existing economic and spatial terms for a complete position."""
        root = self.root_side
        enemy = 1 - root
        difference = state.scores[root] - state.scores[enemy]
        root_region = self.region_size(state, root)
        enemy_region = self.region_size(state, enemy)
        root_moves = len(self.productive_moves_for(state, root))
        enemy_moves = len(self.productive_moves_for(state, enemy))
        center_delta = self.center_value(state.bodies[root][0]) - self.center_value(
            state.bodies[enemy][0]
        )
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
        value += (int(enemy_moves == 0) - int(root_moves == 0)) * 45_000
        value += self.target_race_value(state, root, enemy)
        value += self.star_difference(state, root, enemy)
        return value

    def target_race_value(self, state, root, enemy):
        """Preserve the target race term only when both current heads are known."""
        if self.target is None or not state.bodies[root] or not state.bodies[enemy]:
            return 0
        return (
            self.manhattan(state.bodies[enemy][0], self.target)
            - self.manhattan(state.bodies[root][0], self.target)
        ) * 250

    def star_difference(self, state, root, enemy):
        """Skip optional protection calculations when neither side has a star opportunity."""
        if not (state.stars or state.invincible_a or state.invincible_b):
            return 0
        return (
            self.star_position_value(state, root) - self.star_position_value(state, enemy)
        ) * 22

    def star_position_value(self, state, side):
        """Price visible protection/ram opportunities on the existing point scale."""
        body, rival = state.bodies[side], state.bodies[1 - side]
        if not all((body, rival)):
            return 0
        turns = max(0, (state.remaining_moves + (side == state.side)) // 2)
        duration = min(state.invincibility(side), turns)
        nearest = min(self.manhattan(body[0], cell) for cell in rival)
        # Protection can avoid a first-crash wipe; ramming exposes the detached
        # cells' real 100*mult reward. Distance and the actual own-turn horizon
        # bound both rather than granting a fixed star bonus.
        benefit_base = self.protection_benefit_base(state, side, rival)
        points = benefit_base * duration // (STAR_TURNS * max(1, nearest))
        best = self.best_star_race(state, side, turns, benefit_base, nearest)
        return points + best

    def best_star_race(self, state, side, turns, benefit_base, nearest):
        """Find the existing best star benefit, or zero with no remaining opportunity."""
        if not state.stars or turns <= 0:
            return 0
        return max(
            self.star_race_value(state, side, star, turns, benefit_base, nearest)
            for star in state.stars
        )

    @staticmethod
    def protection_benefit_base(state, side, rival):
        """Keep the accepted first-crash and detached-food economic scale."""
        history = (state.crashed_a, state.crashed_b)[side]
        loss = max(0, state.scores[side]) if history is False else 500
        return loss + max(0, len(rival) - 1) * 100 * state.multipliers[side]

    def star_race_value(self, state, side, star, turns, benefit_base, nearest):
        """Value an attainable star within the actor's remaining turn horizon."""
        ours = self.manhattan(state.bodies[side][0], star)
        theirs = self.manhattan(state.bodies[1 - side][0], star)
        if not self.can_win_star(state, side, ours, theirs, turns):
            return 0
        protected_after = max(0, state.invincibility(side) - ours)
        extra = max(0, min(STAR_TURNS, turns - ours) - protected_after)
        benefit = benefit_base * extra // (STAR_TURNS * max(1, nearest))
        return (100 * state.multipliers[side] + benefit) // max(1, ours)

    @staticmethod
    def can_win_star(state, side, ours, theirs, turns):
        """Resolve ties by alternating turn order, with frenzy stars unopposed."""
        if ours > turns:
            return False
        if state.star_frenzy:
            return True
        return ours < theirs or (ours == theirs and side == state.side)

    def future_food_base_points(self, state, side):
        """Estimate existing sequence income within the remaining turn horizon."""
        if not state.food_values or not state.next_food_digit:
            return FOOD_SCORE
        ordered = self.ordered_food_digits(state)
        turns_left = max(0, (state.remaining_moves + (1 if side == state.side else 0)) // 2)
        captures = min(len(ordered), max(1, turns_left // 8))
        return sum(food_points(digit) for digit in ordered[:captures])

    @staticmethod
    def ordered_food_digits(state):
        """Use the cyclic sequence, including its 9-to-1 wrap."""
        present = {digit for _cell, digit in state.food_values}
        sequence = (cyclic_digit(state.next_food_digit, offset) for offset in range(9))
        return [digit for digit in sequence if digit in present]

    def productive_moves_for(self, state, side):
        """Count movement escapes while excluding protected stationary actions."""
        moves = self.legal_moves_for(state, side)
        body = state.bodies[side]
        if not body:
            return ()
        wrong = self.wrong_food_cells(state, side)
        walls = set(state.walls)
        return tuple(
            direction
            for direction in moves
            if self.productive_target(state, side, direction, walls, wrong)
        )

    def productive_target(self, state, side, direction, walls, wrong):
        """Reject stationary protection actions when counting productive escapes."""
        body = state.bodies[side]
        target = self.target_index(body[0], direction)
        if target in walls or target in wrong:
            return False
        if state.invincibility(side):
            return target is not None and target not in body
        return True

    @staticmethod
    def wrong_food_cells(state, side):
        """Wrong digits block ordinary movement but cost nothing while invincible."""
        if state.invincibility(side):
            return ()
        return tuple(cell for cell, _digit in state.food_values if cell not in state.food)

    def legal_moves_for(self, state, side):
        """Evaluate legal directions for either actor without changing other state."""
        if side == state.side:
            return self.legal_moves(state)
        swapped = replace(state, side=side)
        return self.legal_moves(swapped)

    def expand(self, bits):
        """Expand a frontier to four or six neighbors without wrapping across faces."""
        left = (bits & ~self.left_edge) >> 1
        right = (bits & ~self.right_edge) << 1
        up = (bits & ~self.row_edge) >> self.cols
        down = ((bits & ~self.row_end) << self.cols) & self.full_mask
        planar = left | right | up | down
        if self.levels == 1:
            return planar
        stride = self.rows * self.cols
        return planar | (bits >> stride) | ((bits << stride) & self.full_mask)

    def region_size(self, state, side):
        """Count reachable confirmed cells using bitset frontier expansion."""
        body = state.bodies[side]
        if not body:
            return 0
        start = 1 << body[0]
        wrong_food = self.wrong_food_cells(state, side)
        blocked = (self.occupied_mask(state) | self.mask(wrong_food)) & ~start
        if state.partial:
            known = self.observed_mask(state)
            blocked |= self.full_mask & ~known
        seen = start
        frontier = start
        while frontier:
            frontier = self.expand(frontier) & ~blocked & ~seen
            seen |= frontier
        return seen.bit_count()

    def observed_mask(self, state):
        """Bound partial reachability to cells in the supplied three planes."""
        return self.mask(self.index(cell) for cell, _symbol in state.visibility_key[0])

    def center_value(self, index):
        """Return the accepted integer-rounded centre preference."""
        return -int(self.center_distance(index))

    def center_distance(self, index):
        """Unrounded geometric distance used by move ordering and evaluation."""
        return sum(
            abs(axis - (size - 1) / 2) for axis, size in zip(self.cell(index), self.dimensions)
        )

    def manhattan(self, first, second):
        """Measure coordinate distance across every active axis."""
        return manhattan(self.cell(first), self.cell(second))
